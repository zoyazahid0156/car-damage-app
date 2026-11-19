from contextlib import asynccontextmanager
from pathlib import Path
import base64
import json
import os
import secrets
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import damage_pipeline as pipeline
import price_scraper as ps          # your Selenium / PakWheels code (unchanged)


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "api_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS

# ---- settings you can change with environment variables (Railway: service > Variables)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))     # biggest single file accepted
APP_USER = os.environ.get("APP_USER", "surveyor")             # login name (only used if APP_PASSWORD is set)
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")             # set this to put the whole site behind a login


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = None
    app.state.model_load_error = None

    try:
        print("Loading car part and damage models...")
        app.state.models = pipeline.load_models()
        print("Both models loaded successfully.")
    except Exception:
        app.state.model_load_error = traceback.format_exc()
        print("MODEL LOADING FAILED:")
        print(app.state.model_load_error)

    ps.cleanup_stale_temp_dirs()
    yield


app = FastAPI(
    title="Car Damage Assessment + Part Prices API",
    version="2.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def require_login(request, call_next):
    """If APP_PASSWORD is set, every page, API call and image needs that login (the browser shows a login box)."""
    if not APP_PASSWORD:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            if (secrets.compare_digest(user.encode(), APP_USER.encode())
                    and secrets.compare_digest(password.encode(), APP_PASSWORD.encode())):
                return await call_next(request)
        except Exception:
            pass
    return Response("Login required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Car damage assessment"'})


app.mount(
    "/files",
    StaticFiles(directory=str(RUNS_DIR)),
    name="files",
)


# ----------------------------------------------------------------- part prices (Selenium)
# Damage-model part names -> the words people use when listing that part on PakWheels.
PART_SEARCH = {
    "hood": "bonnet", "bonnet": "bonnet", "trunk_lid": "trunk", "trunk": "trunk",
    "taillight": "tail light", "tail_light": "tail light", "headlight": "headlight",
    "wheel": "alloy rim", "tire": "tyre", "side_mirror": "side mirror",
    "windshield": "windshield", "front_windshield": "windshield", "rear_windshield": "rear windshield",
    "license_plate": "number plate", "front_bumper": "front bumper", "rear_bumper": "rear bumper",
    "front_door": "front door", "rear_door": "rear door",
    "front_fender": "fender", "rear_fender": "fender", "fender": "fender",
}


def search_term_for(part):
    """'front_bumper' -> 'front bumper', 'hood' -> 'bonnet'. Unknown names: underscores become spaces."""
    key = "_".join(part.lower().split()).replace("-", "_")
    return PART_SEARCH.get(key, " ".join(key.split("_")))


@app.get("/prices")
def prices(
    car: str = Query(..., min_length=2, max_length=60, description="e.g. honda civic 2017"),
    part: str = Query(..., min_length=2, max_length=60,
                      description="any part: front bumper, headlight, bonnet ... (damage names like hood also work)"),
):
    """Runs your Selenium scraper for ONE part. It takes roughly 20-60 seconds."""
    car = ps.clean_input(car, "car")
    part = ps.clean_input(search_term_for(part), "part")
    cfg = ps.build_config(car, part)

    if not ps.BROWSER_SLOTS.acquire(timeout=ps.QUEUE_TIMEOUT):
        raise HTTPException(503, "Server is busy, please try again shortly.")
    try:
        items = ps.fetch_prices(cfg)
    except ps.TimeoutException:
        raise HTTPException(504, "PakWheels took too long to respond.")
    except ps.WebDriverException:
        print(traceback.format_exc())
        raise HTTPException(502, "Browser error while searching PakWheels (is Chrome installed?).")
    finally:
        ps.BROWSER_SLOTS.release()

    message = None
    if items is None:
        items, message = [], "No listings found on PakWheels (or the page did not load)."
    elif not items:
        message = "Listings were found, but none were a complete part."
    return {"car": car, "part": part, "search": cfg["search"], "count": len(items),
            "results": items, "message": message}


@app.get("/health")
def health():
    models = getattr(app.state, "models", None)

    return {
        "status": "ok" if models is not None else "model_load_error",
        "models_loaded": models is not None,
        "model_load_error": getattr(app.state, "model_load_error", None),
        "part_model": str(pipeline.PART_MODEL),
        "part_model_exists": Path(pipeline.PART_MODEL).is_file(),
        "damage_model": str(pipeline.DAMAGE_MODEL),
        "damage_model_exists": Path(pipeline.DAMAGE_MODEL).is_file(),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/inspect")
async def inspect(
    files: list[UploadFile] = File(...),
    mode: str = Form("claim"),
    make: str = Form(""),
    model: str = Form(""),
    year: str = Form(""),
    variant: str = Form(""),
    min_frames: int = Form(0),
    show_all_parts: bool = Form(False),
    damage_conf: float = Form(0.15),
):
    models = getattr(app.state, "models", None)

    if models is None:
        error = getattr(app.state, "model_load_error", None)
        raise HTTPException(
            status_code=503,
            detail="Models failed to load. Check /health. "
                   + (error or "No exception details available."),
        )

    if mode not in {"claim", "batch"}:
        raise HTTPException(
            status_code=400,
            detail="Mode must be claim or batch.",
        )

    if not 0.02 <= damage_conf <= 0.9:
        raise HTTPException(status_code=400, detail="damage_conf must be between 0.02 and 0.9.")

    if min_frames < 0:
        raise HTTPException(
            status_code=400,
            detail="min_frames cannot be negative.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one image or video.",
        )

    suffixes = [
        Path(f.filename or "").suffix.lower()
        for f in files
    ]

    if any(ext not in ALLOWED_EXTS for ext in suffixes):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPG, PNG, BMP, WEBP, "
                   "MP4, MOV, AVI, MKV, M4V or WEBM.",
        )

    video_files = [
        ext in VIDEO_EXTS for ext in suffixes
    ]

    if any(video_files):
        if len(files) != 1:
            raise HTTPException(
                status_code=400,
                detail="Upload one video OR multiple photos, not both.",
            )

        if mode != "claim":
            raise HTTPException(
                status_code=400,
                detail="Videos must use claim mode.",
            )

    if len(files) > 60:
        raise HTTPException(
            status_code=400,
            detail="Maximum 60 photos per inspection.",
        )

    inspection_id = uuid.uuid4().hex[:12]
    job_dir = RUNS_DIR / inspection_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "assessed_images"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_files = []

        for index, upload in enumerate(files, start=1):
            suffix = Path(upload.filename or "").suffix.lower()
            destination = input_dir / f"upload_{index:03d}{suffix}"

            try:
                total = 0

                with destination.open("wb") as out:
                    while True:
                        chunk = await upload.read(1024 * 1024)

                        if not chunk:
                            break

                        total += len(chunk)

                        if total > MAX_UPLOAD_MB * 1024 * 1024:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Each uploaded file must be {MAX_UPLOAD_MB} MB or smaller.",
                            )

                        out.write(chunk)

                if total == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Uploaded file {index} is empty.",
                    )

                saved_files.append(destination)

            finally:
                await upload.close()

        # A video is passed as a file. Photos are passed as a folder.
        input_path = (
            saved_files[0]
            if video_files[0]
            else input_dir
        )

        vehicle = {
            "make": make.strip(),
            "model": model.strip(),
            "year": year.strip(),
            "variant": variant.strip(),
        }

        try:
            # Matches the actual damage_pipeline.py signature.
            result, annotated_count = await run_in_threadpool(
                pipeline.assess_path,
                str(input_path),
                models,
                mode,
                vehicle,
                min_frames,
                str(output_dir),
                show_all_parts,
                damage_conf,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            print("ASSESSMENT FAILED:")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Assessment failed: {type(exc).__name__}: {exc}",
            ) from exc

        annotated_images = []

        for image_path in sorted(
            output_dir.glob("*_assessed.jpg")
        ):
            annotated_images.append({
                "filename": image_path.name,
                "url": (
                    f"/files/{inspection_id}/assessed_images/"
                    f"{image_path.name}"
                ),
            })

        urls_by_filename = {
            image["filename"]: image["url"]
            for image in annotated_images
        }

        # Add image URLs to the detailed pipeline result.
        for item in result.get("per_image", []):
            filename = item.get("annotated_image")

            if filename:
                item["annotated_image_url"] = (
                    urls_by_filename.get(filename)
                )

        report_path = job_dir / "assessment.json"
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "inspection_id": inspection_id,
            "result": result,
            "annotated_images_saved": annotated_count,
            "annotated_images": annotated_images,
            "json_url": f"/files/{inspection_id}/assessment.json",
            "review_note": (
                "Severity is an estimate. A surveyor should review "
                "the damage and any suspected missing parts."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Car Damage Assessment</title>
<style>
body{font-family:Arial,sans-serif;max-width:1000px;margin:25px auto;padding:0 16px;color:#18212f}
.panel{border:1px solid #d9dee7;border-radius:12px;padding:16px;margin:14px 0}
.grid,.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
label{display:block;font-weight:bold;margin:12px 0 5px}
input,select,button{font:inherit;box-sizing:border-box;padding:10px;border-radius:7px}
input:not([type=checkbox]),select{width:100%;border:1px solid #b9c2d0}
button{margin-top:16px;background:#1d4ed8;color:white;border:0;cursor:pointer}
button.small{margin:8px 0 0;padding:6px 12px;font-size:14px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d9dee7;padding:6px;text-align:left}td.n,th.n{text-align:right}
.muted{color:#596579}#status{white-space:pre-wrap;overflow-wrap:anywhere}
.card{border:1px solid #d9dee7;border-radius:9px;padding:12px;overflow-wrap:anywhere}
img{width:100%;height:auto;border-radius:7px;margin-top:8px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;overflow:auto}
</style>
</head>
<body>
<h1>Car Damage Assessment</h1>
<p class="muted">Upload several photos of one car or one video.
View detected damaged parts, severity, counts and annotated images.</p>

<form id="form" class="panel">
<label>Input type</label>
<select id="kind">
<option value="photos">Multiple photos of the same car</option>
<option value="video">One video</option>
<option value="batch">Photos of different cars (batch)</option>
</select>

<div class="grid">
<div><label>Make (optional)</label><input name="make" placeholder="Honda"></div>
<div><label>Model (optional)</label><input name="model" placeholder="Civic"></div>
<div><label>Year (optional)</label><input name="year" placeholder="2017"></div>
<div><label>Variant (optional)</label><input name="variant" placeholder="Oriel"></div>
</div>

<label>Select photos or video</label>
<input id="files" type="file" multiple required
accept="image/jpeg,image/png,image/webp,image/bmp">

<p class="muted" id="hint">Choose multiple photos of one car.</p>

<label>Damage sensitivity (lower finds more damage, more false alarms)</label>
<input name="damage_conf" type="number" step="0.01" min="0.02" max="0.9" value="0.15">

<label><input type="checkbox" name="show_all_parts" value="true">
Also show detected undamaged parts</label>

<button id="submit" type="submit">Assess damage</button>
</form>

<div id="status" role="status" aria-live="polite"></div>
<div id="results"></div>

<section class="panel" id="pricePanel">
<h2>Part price (PakWheels)</h2>
<p class="muted">Click "Get price" on a damaged part above, or type any part. Searching takes about 20-60 seconds.</p>
<div class="grid">
<div><label for="priceCar">Car</label><input id="priceCar" placeholder="honda civic 2017"></div>
<div><label for="pricePart">Part</label><input id="pricePart" placeholder="front bumper"></div>
</div>
<button id="priceBtn" type="button">Find price</button>
<div id="priceStatus" role="status" aria-live="polite" class="muted"></div>
<div id="priceOut"></div>
</section>

<script>
const kind=document.getElementById('kind');
const files=document.getElementById('files');
const hint=document.getElementById('hint');
const status=document.getElementById('status');
const results=document.getElementById('results');

function updateInput(){
  files.value='';
  files.multiple=kind.value!=='video';
  files.accept=kind.value==='video'
    ?'.mp4,.mov,.avi,.mkv,.m4v,.webm'
    :'image/jpeg,image/png,image/webp,image/bmp';
  hint.textContent=kind.value==='video'
    ?'Choose one video of the car.'
    :kind.value==='batch'
      ?'Each photo is assessed independently.'
      :'Choose multiple angles of the same car.';
}
kind.addEventListener('change',updateInput);
updateInput();

function esc(v){
  return String(v??'').replace(/[&<>"']/g,c=>({
    '&':'&amp;','<':'&lt;','>':'&gt;',
    '"':'&quot;',"'":'&#39;'
  }[c]));
}

function partCard(p){
  return `<div class="card">
    <strong>${esc(p.part)}</strong>
    <p>Severity: ${esc(p.severity||'unknown')}</p>
    <p>Damage type: ${esc(p.damage_type||'unknown')}</p>
    <p>Count: ${esc(p.count??1)}</p>
    <p>Damaged area: ${esc(p.damage_area_pct??0)}%</p>
    ${p.confidence==='possible'
      ?`<p style="color:#b45309"><b>Possible</b> &mdash; damage seen in only ${esc(p.frames_with_damage)} of ${esc(p.frames_seen)} frames. Please check this spot.</p>`
      :(p.frames_seen>1?`<p class="muted">Seen damaged in ${esc(p.frames_with_damage)} of ${esc(p.frames_seen)} frames.</p>`:'')}
    <button type="button" class="small price-btn" data-part="${esc(p.part)}">Get price</button>
  </div>`;
}

document.getElementById('form').addEventListener('submit',async e=>{
  e.preventDefault();

  if(!files.files.length){
    status.textContent='Please select files first.';
    return;
  }

  const selected=[...files.files];
  const isVideo=kind.value==='video';

  if(isVideo && selected.length!==1){
    status.textContent='Choose exactly one video.';
    return;
  }

  if(!isVideo && selected.some(f=>f.type.startsWith('video/'))){
    status.textContent='Choose photos for this input type.';
    return;
  }

  const vf=e.currentTarget.elements;   // read the form BEFORE any await (currentTarget is cleared afterwards)
  const carText=[vf.make.value,vf.model.value,vf.year.value].map(x=>x.trim()).filter(Boolean).join(' ');
  const fd=new FormData();
  fd.append('mode',kind.value==='batch'?'batch':'claim');
  for(const key of ['make','model','year','variant']){
    fd.append(key,e.currentTarget.elements[key].value);
  }
  fd.append('damage_conf',e.currentTarget.elements.damage_conf.value||'0.15');
  fd.append('show_all_parts',
    e.currentTarget.elements.show_all_parts.checked?'true':'false');

  selected.forEach(f=>fd.append('files',f));

  status.textContent='Analysing. Video processing may take some time...';
  results.innerHTML='';
  document.getElementById('submit').disabled=true;

  try{
    const response=await fetch('/inspect',{method:'POST',body:fd});
    const data=await response.json();

    if(!response.ok){
      const detail=data.detail;
      throw new Error(typeof detail==='string'?detail:JSON.stringify(detail));
    }

    const r=data.result;
    let html='';

    if(r.mode==='claim'){
      const parts=r.damaged_parts||[];
      const counts=r.severity_counts||{};

      html+=`<section class="panel"><h2>Summary</h2>
        <p>Damaged part types: ${parts.length}
        (${r.confirmed_parts??parts.length} confirmed, ${r.possible_parts??0} possible)</p>
        <p>Minor: ${counts.minor||0} |
        Moderate: ${counts.moderate||0} |
        Severe: ${counts.severe||0}</p>
        <div class="cards">${parts.map(partCard).join('')}</div></section>`;

      if(r.suspected_missing_parts?.length){
        html+='<section class="panel"><h2>Suspected missing parts</h2><p>Confirm these manually.</p>';
        r.suspected_missing_parts.forEach(p=>{
          html+=`<p>${esc(p.part)} — ${esc(p.status)}</p>`;
        });
        html+='</section>';
      }
    }else{
      html+=`<section class="panel"><h2>Batch summary</h2>
        <p>${r.images_with_damage} of ${r.images_analysed}
        images show detected damage.</p></section>`;

      (r.images||[]).forEach(im=>{
        html+=`<section class="panel"><h3>${esc(im.image)}</h3>
          <p>Detected damaged part types: ${(im.damaged_parts||[]).length}</p>
          <div class="cards">${(im.damaged_parts||[]).map(partCard).join('')}</div>
        </section>`;
      });
    }

    html+='<section class="panel"><h2>Annotated images</h2><div class="cards">';
    if(!data.annotated_images.length){
      html+='<p>No annotated images were saved.</p>';
    }

    data.annotated_images.forEach(im=>{
      html+=`<div class="card"><strong>${esc(im.filename)}</strong>
        <a href="${esc(im.url)}" target="_blank" rel="noopener">
        <img loading="lazy" src="${esc(im.url)}" alt="Annotated damage"></a>
        <p><a href="${esc(im.url)}" target="_blank" rel="noopener">Open image</a></p>
      </div>`;
    });

    html+='</div><p><a href="'+esc(data.json_url)+
      '" target="_blank" rel="noopener">Open full JSON report</a></p></section>';

    html+='<section class="panel"><p class="muted">'+
      esc(data.review_note)+'</p></section>';

    results.innerHTML=html;
    if(carText) document.getElementById('priceCar').value=carText;
    status.textContent='Analysis complete. Inspection ID: '+data.inspection_id;

  }catch(err){
    status.textContent='Error: '+err.message+
      '\nCheck /health and the VS Code terminal.';
  }finally{
    document.getElementById('submit').disabled=false;
  }
});

const priceStatus=document.getElementById('priceStatus');
const priceOut=document.getElementById('priceOut');
const priceBtn=document.getElementById('priceBtn');
const nice=s=>String(s||'').replace(/_/g,' ');
const pkr=n=>'PKR '+Number(n).toLocaleString('en-US');

async function findPrice(){
  const car=document.getElementById('priceCar').value.trim();
  const part=document.getElementById('pricePart').value.trim();
  if(car.length<2||part.length<2){priceStatus.textContent='Enter the car and the part.';return;}
  priceBtn.disabled=true;priceOut.innerHTML='';
  priceStatus.textContent='Searching PakWheels for "'+part+'" ... this takes 20-60 seconds.';
  try{
    const r=await fetch('/prices?car='+encodeURIComponent(car)+'&part='+encodeURIComponent(part));
    const d=await r.json();
    if(!r.ok){const x=d.detail;throw new Error(typeof x==='string'?x:JSON.stringify(x));}
    priceStatus.textContent='';
    if(!d.results.length){priceOut.innerHTML='<p>'+esc(d.message||'No prices found.')+'</p>';return;}
    const v=d.results.map(x=>x.price_pkr).sort((a,b)=>a-b);
    const med=v.length%2?v[(v.length-1)/2]:Math.round((v[v.length/2-1]+v[v.length/2])/2);
    priceOut.innerHTML='<p><b>'+d.count+'</b> listing(s) for <b>'+esc(d.search)+'</b> &mdash; lowest '+pkr(v[0])+
      ', typical '+pkr(med)+', highest '+pkr(v[v.length-1])+'</p><table><tr><th>Listing</th><th class="n">Price</th></tr>'+
      d.results.map(x=>'<tr><td>'+esc(x.name)+'</td><td class="n">'+pkr(x.price_pkr)+'</td></tr>').join('')+'</table>';
  }catch(err){
    priceStatus.textContent='Error: '+err.message;
  }finally{priceBtn.disabled=false;}
}
priceBtn.addEventListener('click',findPrice);
results.addEventListener('click',ev=>{
  const b=ev.target.closest('.price-btn');
  if(!b)return;
  document.getElementById('pricePart').value=nice(b.dataset.part);
  document.getElementById('pricePanel').scrollIntoView({behavior:'smooth'});
  findPrice();
});
</script>
</body>
</html>
"""