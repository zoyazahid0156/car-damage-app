import json
import os
import re
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.pakwheels.com/accessories-spare-parts/"
OUTPUT_FILE = "car_parts.json"
HEADLESS = True  # False = show the Chrome window again

PARTS = {
    "front_bumper": {
        "search": " toyota corolla 2015 front bumper",
        "phrase": "front bumper",
        "category": "bumpers",
    },
    }


# If the title contains any of these it is a sub-part, not the full part
EXCLUDE_WORDS = [
    "spacer", "grille", "grill", "trim", "splitter", "bracket", "retainer",
    "support", "clip", "lip", "guard", "cover", "garnish", "molding",
    "moulding", "reflector", "fog", "light", "lamp", "sticker", "stay",
    "bolt", "screw", "net", "diffuser", "protector", "led", "sill", "plate","bulb", "eyebrow", "trim", "strip", "chrome", "grille", "grill",
    "cover", "garnish", "bracket", "clip", "washer", "nozzle",
    "wiper", "sticker", "lens", "seal", "gasket", "switch", "relay",
    "harness", "wiring", "ballast", "kit", "converter", "protector",
    "guard", "tint", "film",
]

SEARCH_BOX = '//*[@id="search_key"]'
SEARCH_BTN = '//*[@id="ad-listings-search-btn"]'
RESULT_CSS = "div.col-md-7.grid-style, div.col-md-9.grid-style"

BUTTON_LINES = {"buy now", "add to cart", "save ad"}
# Handles "PKR 9,300" and also "PKR 1.75 lacs" (= 175,000)
PRICE_RE = re.compile(
    r"PKR\s*([\d,]*\.?\d+)\s*(lacs?|lakhs?|crores?|million)?", re.IGNORECASE)
MULTIPLIERS = {"lac": 100_000, "lakh": 100_000, "crore": 10_000_000,
               "million": 1_000_000}


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
    if cfg["category"] not in item["category"].lower():
        return False
    words = list(cfg.get("exclude_words", EXCLUDE_WORDS)) + cfg.get("extra_exclude", [])
    # whole-word match (also plurals): "net" must not match "genuine"
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")(e?s)?\b"
    return not re.search(pattern, name)


def make_driver():
    options = webdriver.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
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
    driver = webdriver.Chrome(options=options)
    if not HEADLESS:
        driver.maximize_window()
    driver.set_page_load_timeout(40)
    return driver


def search_part(driver, wait, cfg):
    """Return list of {name, price_pkr} for complete parts, or None if the
    search page did not load."""
    try:
        driver.get(URL)
    except TimeoutException:
        driver.execute_script("window.stop();")

    box = wait.until(EC.element_to_be_clickable((By.XPATH, SEARCH_BOX)))
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
    time.sleep(2)

    # scroll so lazy-loaded results appear
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    blocks = driver.find_elements(By.CSS_SELECTOR, RESULT_CSS)
    print(f"  {len(blocks)} result blocks found")

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


def load_existing(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    data = load_existing(out_path)  # keep parts added earlier

    driver = make_driver()
    wait = WebDriverWait(driver, 30)
    try:
        for key, cfg in PARTS.items():
            print(f"Searching: {cfg['search']}")
            results = search_part(driver, wait, cfg)
            if results is None:
                print("  no results loaded - keeping old data for this part")
                continue
            data[key] = results
            print(f"  kept {len(results)} complete parts")
    finally:
        driver.quit()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}\n")

    for key, items in data.items():
        print(key)
        for r in items:
            print(f"  {r['name']}: PKR {r['price_pkr']:,}")


if __name__ == "__main__":
    main()