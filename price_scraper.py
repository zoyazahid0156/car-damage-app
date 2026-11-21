"""
PakWheels car-part price scraper  (FastAPI + Selenium)

NOTE: main.py (the combined API) imports this file and serves /prices itself, so you only need to run
"uvicorn main:app". You can still run this file alone:  uvicorn price_scraper:app --port 8001

Install:  pip install fastapi uvicorn selenium
Run:      uvicorn main:app --reload
Docs:     http://127.0.0.1:8000/docs

Examples:
  GET /prices?car=toyota corolla 2015&part=front bumper
  GET /prices?car=honda civic&part=headlight
  GET /prices?car=honda civic&part=rear bumper&download=true   (json file)

For every request a headless Chrome is started, the prices are scraped, the
results are written to a TEMPORARY json file, and that file (its whole temp
folder) is deleted as soon as the response has been sent.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from starlette.background import BackgroundTask

log = logging.getLogger("parts_api")

# ----------------------------- settings -----------------------------
URL = "https://www.pakwheels.com/accessories-spare-parts/"
HEADLESS = os.environ.get("HEADLESS", "1") != "0"  # HEADLESS=0 shows Chrome
PAGE_LOAD_TIMEOUT = int(os.environ.get("PAGE_LOAD_TIMEOUT", 90))   # seconds to wait for PakWheels to load
WAIT_SECONDS = int(os.environ.get("SCRAPE_WAIT", 80))             # seconds to wait for the search box / results
SCROLLS = int(os.environ.get("SCRAPE_SCROLLS", 3))                # scrolls to load lazy results (was 5)
CACHE_TTL = float(os.environ.get("PRICE_CACHE_HOURS", 24)) * 3600 # same search again = instant, no browser
_CACHE = {}                                                       # (search, phrase) -> (time, items)
MAX_BROWSERS = 2          # Chrome windows running at the same time
QUEUE_TIMEOUT = 120        # seconds a request waits for a free browser
TEMP_PREFIX = "parts_prices_"
STALE_AFTER_SECONDS = 3600

SEARCH_BOX = '//*[@id="search_key"]'
SEARCH_BTN = '//*[@id="ad-listings-search-btn"]'
RESULT_CSS = "div.col-md-7.grid-style, div.col-md-9.grid-style"

BUTTON_LINES = {"buy now", "add to cart", "save ad"}
# Handles "PKR 9,300" and also "PKR 1.75 lacs" (= 175,000)
PRICE_RE = re.compile(
    r"PKR\s*([\d,]*\.?\d+)\s*(lacs?|lakhs?|crores?|million)?", re.IGNORECASE)
MULTIPLIERS = {"lac": 100_000, "lakh": 100_000, "crore": 10_000_000,
               "million": 1_000_000}

# If a title contains any of these it is a sub-part, not the complete part
EXCLUDE_WORDS = [
    "spacer", "grille", "grill", "trim", "splitter", "bracket", "retainer",
    "support", "clip", "lip", "guard", "cover", "garnish", "molding",
    "moulding", "reflector", "fog", "light", "lamp", "sticker", "stay",
    "bolt", "screw", "net", "diffuser", "protector", "led", "sill", "plate",
    "bulb", "eyebrow", "strip", "chrome", "washer", "nozzle", "wiper",
    "lens", "seal", "gasket", "switch", "relay", "harness", "wiring",
    "ballast", "kit", "converter", "tint", "film",
]

# Parts we have tuned. Anything else uses the generic rules (see build_config).
# "exclude_words" REPLACES EXCLUDE_WORDS for that part.
PART_PRESETS = {
    "front bumper": {"category": "bumpers"},
    "rear bumper": {"category": "bumpers"},
    "headlight": {
        "category": "headlights",
        "exclude_words": [
            "bulb", "eyebrow", "trim", "strip", "chrome", "grille", "grill",
            "cover", "garnish", "bracket", "clip", "washer", "nozzle",
            "wiper", "sticker", "lens", "seal", "gasket", "switch", "relay",
            "harness", "wiring", "ballast", "kit", "converter", "protector",
            "guard", "tint", "film",
        ],
    },
}

BROWSER_SLOTS = threading.BoundedSemaphore(MAX_BROWSERS)
SAFE_TEXT = re.compile(r"^[A-Za-z0-9 .\-/&+()]+$")


# ------------------------- scraping (your logic) -------------------------
def to_pkr(number, unit):
    value = float(number.replace(",", ""))
    unit = (unit or "").lower().rstrip("s")
    return int(round(value * MULTIPLIERS.get(unit, 1)))


def get_lines(el):
    raw = el.text or el.get_attribute("textContent") or ""
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def parse_block(lines):
    """Return dict(name, price, category) or None."""
    if not lines:
        return None
    name = lines[0]

    price = None
    price_line = None
    for ln in lines:
        m = PRICE_RE.findall(ln)
        if m:
            # "PKR 12,153 PKR 14,820" -> first one is the current (sale) price
            price = to_pkr(*m[0])
            price_line = ln
            break

    category = ""
    for ln in lines[1:]:
        if ln == price_line or ln.lower() in BUTTON_LINES:
            continue
        category = ln
        break

    return {"name": name, "price": price, "category": category}


def is_complete_part(item, cfg):
    name = item["name"].lower()
    if cfg["phrase"] not in name:
        return False
    if cfg["category"] not in item["category"].lower():  # "" always passes
        return False
    words = list(cfg.get("exclude_words", EXCLUDE_WORDS)) + cfg.get("extra_exclude", [])
    # whole-word match (also plurals): "net" must not match "genuine"
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")(e?s)?\b"
    return not re.search(pattern, name)


def make_driver():
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    # pictures are not needed for prices; skipping them makes the page load much faster
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    if os.environ.get("CHROME_NO_SANDBOX") == "1":       # needed when Chrome runs as root inside Docker (Railway)
        options.add_argument("--no-sandbox")
    if os.environ.get("CHROME_BIN"):                      # Chrome installed by the Dockerfile
        options.binary_location = os.environ["CHROME_BIN"]
    if HEADLESS:
        options.add_argument("--headless=new")
        # big window so the desktop layout (col-md-*) is rendered
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
    if os.environ.get("CHROMEDRIVER"):                    # chromedriver installed by the Dockerfile
        from selenium.webdriver.chrome.service import Service
        driver = webdriver.Chrome(options=options, service=Service(os.environ["CHROMEDRIVER"]))
    else:
        driver = webdriver.Chrome(options=options)
    if not HEADLESS:
        driver.maximize_window()
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def search_part(driver, wait, cfg):
    """Return list of {name, price_pkr} for complete parts, or None if the
    search page showed no results."""
    try:
        driver.get(URL)
    except TimeoutException:
        driver.execute_script("window.stop();")

    try:
        box = wait.until(EC.element_to_be_clickable((By.XPATH, SEARCH_BOX)))
    except TimeoutException:
        # say WHAT the browser saw, so a block page (Cloudflare / "Access denied") can be told from a slow page
        raise TimeoutException(f"search box not found; page title was {driver.title!r}")
    box.clear()
    box.send_keys(cfg["search"])
    try:
        driver.find_element(By.XPATH, SEARCH_BTN).click()
    except Exception:
        box.send_keys(Keys.ENTER)

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, RESULT_CSS)))
    except TimeoutException:
        return None
    time.sleep(1)

    # scroll so lazy-loaded results appear
    for _ in range(SCROLLS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.7)

    blocks = driver.find_elements(By.CSS_SELECTOR, RESULT_CSS)
    log.info("%d result blocks for '%s'", len(blocks), cfg["search"])

    results, seen = [], set()
    for block in blocks:
        item = parse_block(get_lines(block))
        if not item or item["price"] is None:
            continue
        if not is_complete_part(item, cfg):
            continue
        key = (item["name"], item["price"])
        if key in seen:
            continue
        seen.add(key)
        results.append({"name": item["name"], "price_pkr": item["price"]})
    return results


def fetch_prices(cfg):
    """Start a fresh headless Chrome, scrape, always close it. The same search within PRICE_CACHE_HOURS
    is answered from memory (no browser)."""
    key = (cfg["search"].lower(), cfg["phrase"])
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    driver = make_driver()
    try:
        items = search_part(driver, WebDriverWait(driver, WAIT_SECONDS), cfg)
    finally:
        driver.quit()
    if items:                                   # only remember real results, never failures
        _CACHE[key] = (time.time(), items)
    return items


# ----------------------- turn user input into a config -----------------------
def singular(text):
    words = text.split()
    last = words[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    return " ".join(words)


def build_config(car, part):
    part_norm = " ".join(part.lower().split())
    phrase = singular(part_norm)  # "headlights" -> "headlight" (still matches plurals)

    preset = PART_PRESETS.get(phrase)
    if preset:
        cfg = dict(preset)
    else:
        # Generic part: no category check, and never exclude a word that is
        # part of the part name itself (e.g. "fog light" must keep "light").
        part_words = set(re.findall(r"[a-z0-9]+", phrase))
        cfg = {
            "category": "",
            "exclude_words": [w for w in EXCLUDE_WORDS if w not in part_words],
        }
    cfg["search"] = f"{car} {part_norm}"
    cfg["phrase"] = phrase
    return cfg


def clean_input(value, field):
    value = " ".join(value.split())
    if not (2 <= len(value) <= 60) or not SAFE_TEXT.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field}: use 2-60 letters, numbers or spaces.")
    return value


# ------------------------------- temp files -------------------------------
def cleanup_stale_temp_dirs():
    """Remove leftovers from a crashed run (only our own old temp folders)."""
    root = tempfile.gettempdir()
    now = time.time()
    for name in os.listdir(root):
        if not name.startswith(TEMP_PREFIX):
            continue
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) > STALE_AFTER_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app):
    cleanup_stale_temp_dirs()
    yield


# ---------------------------------- API ----------------------------------
app = FastAPI(
    title="PakWheels Car Part Prices",
    lifespan=lifespan,
    # hides the "Schemas" section at the bottom of /docs
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


class PartItem(BaseModel):
    name: str
    price_pkr: int


class PricesResponse(BaseModel):
    car: str
    part: str
    search: str
    count: int
    results: List[PartItem]
    message: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/prices", response_model=PricesResponse)
def get_prices(
    car: str = Query(..., min_length=2, max_length=60,
                     description="e.g. toyota corolla 2015"),
    part: str = Query(..., min_length=2, max_length=60,
                      description="e.g. front bumper, rear bumper, headlight"),
    download: bool = Query(False, description="Return the temporary json file"),
):
    car = clean_input(car, "car")
    part = clean_input(part, "part")
    cfg = build_config(car, part)

    # Wait for a free browser slot
    if not BROWSER_SLOTS.acquire(timeout=QUEUE_TIMEOUT):
        raise HTTPException(503, "Server is busy, please try again shortly.")
    try:
        items = fetch_prices(cfg)
    except TimeoutException:
        raise HTTPException(504, "PakWheels took too long to respond.")
    except WebDriverException:
        log.exception("Browser error")
        raise HTTPException(502, "Browser error while searching PakWheels.")
    finally:
        BROWSER_SLOTS.release()

    message = None
    if items is None:
        items = []
        message = "No listings found on PakWheels (or the page did not load)."
    elif not items:
        message = "Listings were found, but none were a complete part."

    payload = {
        "car": car,
        "part": part,
        "search": cfg["search"],
        "count": len(items),
        "results": items,
        "message": message,
    }

    # Temporary json file: created per search, deleted right after use
    tmpdir = tempfile.mkdtemp(prefix=TEMP_PREFIX)
    handed_off = False
    try:
        path = os.path.join(tmpdir, "prices.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        if download:
            handed_off = True  # deleted by the background task after sending
            return FileResponse(
                path,
                media_type="application/json",
                filename="prices.json",
                background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True),
            )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if not handed_off:
            shutil.rmtree(tmpdir, ignore_errors=True)


# Lets you start the server with "python main.py" or your editor's Run button
if __name__ == "__main__":
    import uvicorn

    print("Server starting...")
    print("Open in browser: http://127.0.0.1:8000/docs")
    print("Example: http://127.0.0.1:8000/prices?car=toyota corolla 2015&part=front bumper")
    uvicorn.run(app, host="127.0.0.1", port=8000)