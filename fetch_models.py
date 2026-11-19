"""
Download the model files (best.pt) when the server starts, because they are too big for GitHub.

Set these in Railway (service > Variables). A model is only downloaded if its file is not there yet.

  PART_MODEL_URL       direct download link of the part detector best.pt        (required)
  DAMAGE_MODEL_URL     direct download link of the CarDD / merged damage best.pt (required)
  DAMAGE_MODEL_2_URL   direct link of the optional second damage model (26k)     (optional)
  MODEL_DOWNLOAD_TOKEN access token if the links are private (Hugging Face "read" token) (optional)

Example link (Hugging Face):  https://huggingface.co/YOUR-NAME/car-damage-models/resolve/main/part_best.pt

Run by hand to test:  python fetch_models.py
"""
import os
import sys
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# These paths must match damage_pipeline.py (PART_MODEL, DAMAGE_MODEL and the default second model).
MODELS = [
    ("PART_MODEL_URL", ROOT / "runs/detect/detector_results_17/car_parts_new/weights/best.pt", True),
    ("DAMAGE_MODEL_URL", ROOT / "damage_results/damage_seg/weights/best.pt", True),
    ("DAMAGE_MODEL_2_URL", ROOT / "damage_26" / "best_26k.pt", False),
]
MIN_BYTES = 1_000_000          # anything smaller is an error page, not a model


class KeepAuthOnlyOnSameHost(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects the download to another host (a CDN with a signed link).
    Sending the token there makes the CDN answer 401/400, so the token is dropped on a redirect to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def download(url, dest, token):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "car-damage-server"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    opener = urllib.request.build_opener(KeepAuthOnlyOnSameHost)
    with opener.open(request, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last_print = 0, 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done - last_print >= 50 * (1 << 20):
                last_print = done
                print(f"    {done / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else ""), flush=True)
    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink()
        raise RuntimeError(f"Downloaded file is only a few bytes - the link or token is wrong: {url}")
    tmp.replace(dest)


def main():
    token = os.environ.get("MODEL_DOWNLOAD_TOKEN", "").strip()
    failed = False
    for variable, dest, required in MODELS:
        url = os.environ.get(variable, "").strip()
        if dest.exists() and dest.stat().st_size >= MIN_BYTES:
            print(f"OK (already there): {dest.relative_to(ROOT)}")
            continue
        if not url:
            if required:
                print(f"MISSING: {dest.relative_to(ROOT)} - set the {variable} variable.")
                failed = True
            else:
                print(f"Skipping the optional model ({variable} not set).")
            continue
        print(f"Downloading {dest.relative_to(ROOT)} ...", flush=True)
        try:
            download(url, dest, token)
            print(f"  done ({dest.stat().st_size / 1e6:.0f} MB)")
        except Exception as ex:                       # keep the message short; never print the token
            print(f"  FAILED: {type(ex).__name__}: {ex}")
            if required:
                failed = True
    if failed:
        sys.exit("Model files are missing - the server cannot start. Fix the variables above and redeploy.")


if __name__ == "__main__":
    main()
