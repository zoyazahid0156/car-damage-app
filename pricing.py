"""pricing.py - live spare-part prices from PakWheels using Selenium.

Replacement prices come only from live PakWheels search listings. If the live
lookup finds no usable listing, replacement cost is None; there is no CSV price
fallback. Repair costs may come from repair_quotes.csv (shop quotes or explicitly
labelled indicative figures). Labour amounts are placeholders unless configured.

Test a live part price:
    python pricing.py --make Honda --model Civic --year 2018 --part front_bumper --debug

Part prices are seller asking prices and estimates. Use responsibly and respect
PakWheels' terms of use. Requests are spaced by DELAY_SECONDS.
"""
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_FILE = HERE / "price_cache.json"
REPAIR_QUOTES_FILE = HERE / "repair_quotes.csv"
CACHE_DAYS = 7
DELAY_SECONDS = 4
BASE_URL = "https://www.pakwheels.com/accessories-spare-parts/"
SEARCH_INPUT_CSS = "input#search_key[name='search_key']"
TITLE_CSS = "h3"
PRICE_CSS = "div.price-details"

# ---------------------------------------------------------------- what to search for, per part
# terms : search words appended to "<make> <model>"
# must  : listing title must contain at least one of these
# block : listing title must contain none of these (filters out insulators, covers, stickers ...)
# min   : prices below this are treated as accessories, not the real part (PKR) - tune after --debug runs
COMMON_BLOCK = ["insulator", "insulation", "namda", "protector", "sticker", "decal", "mat ", "mats", "clip",
                "liner", "sheet", "badge", "emblem", "logo", "gasket", "hinge", "latch", "cable", "strut",
                "shock", "wiper", "toy", "model car", "remote", "stand", "paint", "polish", "cleaner"]

PARTS = {
    "hood":             {"terms": ["bonnet"], "must": ["bonnet", "hood"], "block": COMMON_BLOCK + ["scoop", "cover"], "min": 8000},
    "front_bumper":     {"terms": ["front bumper"], "must": ["bumper"], "block": COMMON_BLOCK + ["rear", "back", "lip", "guard"], "min": 5000},
    "rear_bumper":      {"terms": ["rear bumper"], "must": ["bumper"], "block": COMMON_BLOCK + ["front", "lip", "guard"], "min": 5000},
    "front_door":       {"terms": ["front door"], "must": ["door"], "block": COMMON_BLOCK + ["glass", "handle", "mirror", "rear", "lock", "trim", "pad", "window", "sill", "visor", "weather", "strip", "speaker"], "min": 8000},
    "rear_door":        {"terms": ["rear door"], "must": ["door"], "block": COMMON_BLOCK + ["glass", "handle", "mirror", "front", "lock", "trim", "pad", "window", "sill", "visor", "weather", "strip", "speaker"], "min": 8000},
    "front_fender":     {"terms": ["fender"], "must": ["fender", "wing"], "block": COMMON_BLOCK + ["flare", "rear", "trim", "mudguard"], "min": 4000},
    "rear_fender":      {"terms": ["quarter panel", "rear fender"], "must": ["fender", "quarter"], "block": COMMON_BLOCK + ["flare", "front", "trim"], "min": 4000},
    "trunk_lid":        {"terms": ["trunk lid", "dicky door"], "must": ["trunk", "dicky", "boot"], "block": COMMON_BLOCK + ["spoiler", "mat", "cover", "handle", "organizer"], "min": 6000},
    "headlight":        {"terms": ["headlight", "head lamp"], "must": ["headlight", "head light", "head lamp", "headlamp"], "block": COMMON_BLOCK + ["bulb", "cover", "washer", "switch", "relay", "wire", "harness", "kit", "drl strip"], "min": 3000},
    "taillight":        {"terms": ["tail light", "tail lamp"], "must": ["tail"], "block": COMMON_BLOCK + ["bulb", "cover", "switch", "wire", "harness", "kit"], "min": 2000},
    "side_mirror":      {"terms": ["side mirror"], "must": ["mirror"], "block": COMMON_BLOCK + ["cover", "lens", "blind", "convex", "interior", "rear view", "rearview", "inside", "camera", "wide", "film", "cap", "indicator"], "min": 1500},
    "windshield_front": {"terms": ["windshield", "windscreen"], "must": ["windshield", "windscreen", "wind screen"], "block": COMMON_BLOCK + ["washer", "sunshade", "shade", "cover", "film", "cleaner", "sun ", "blade", "rubber", "tint", "visor"], "min": 5000},
    "windshield_back":  {"terms": ["rear windshield", "back glass"], "must": ["windshield", "windscreen", "glass"], "block": COMMON_BLOCK + ["front", "washer", "sunshade", "shade", "cover", "film", "blade", "rubber", "tint", "visor"], "min": 3000},
    "front_window":     {"terms": ["door glass"], "must": ["glass"], "block": COMMON_BLOCK + ["film", "tint", "curtain", "visor", "rear", "windshield"], "min": 2000},
    "rear_window":      {"terms": ["door glass"], "must": ["glass"], "block": COMMON_BLOCK + ["film", "tint", "curtain", "visor", "front", "windshield"], "min": 2000},
    "wheel":            {"terms": ["alloy rim", "rim"], "must": ["rim", "wheel", "alloy"], "block": COMMON_BLOCK + ["cover", "cap", "bolt", "nut", "lock", "stud", "spacer", "tyre", "tire", "trim"], "min": 5000},
    # license_plate is not supported by the live listing scraper.
}

# ---------------------------------------------------------------- repair / replace rules
ALWAYS_REPLACE = {"windshield_front", "windshield_back", "headlight", "taillight", "side_mirror",
                  "front_window", "rear_window"}
BODY_PANELS = {"hood", "front_bumper", "rear_bumper", "front_door", "rear_door", "front_fender",
               "rear_fender", "trunk_lid"}

# PLACEHOLDERS (PKR) - only used when include_labour=True
FITTING_PKR = {"body": 6000, "other": 3000}
PAINT_PKR = {"body": 12000, "other": 0}


# ================================================================ repair quotes (shops)
class RepairQuotes:
    """Reads repair_quotes.csv: part, severity, shop, min_pkr, max_pkr, source, note.
    Rows whose shop starts with 'indicative' are public web figures, not shop quotes."""
    FIELDS = ["part", "severity", "shop", "min_pkr", "max_pkr", "source", "note"]

    def __init__(self, path=REPAIR_QUOTES_FILE):
        self.path = Path(path)
        self.rows = []
        if self.path.is_file():
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
                    try:
                        r["min_pkr"], r["max_pkr"] = float(r["min_pkr"]), float(r["max_pkr"])
                    except (KeyError, ValueError):
                        continue
                    self.rows.append(r)

    @staticmethod
    def _indicative(r):
        return r.get("shop", "").lower().startswith("indicative")

    def lookup(self, part, severity, indicative=False):
        rows = [r for r in self.rows if r.get("part", "").lower() == part
                and r.get("severity", "").lower() == severity and self._indicative(r) == indicative]
        if not rows:
            return None
        lo = int(statistics.median(r["min_pkr"] for r in rows))
        hi = int(statistics.median(r["max_pkr"] for r in rows))
        if indicative:
            basis = "INDICATIVE public figure, not a shop quote (" + "; ".join(sorted({r.get("source", "") for r in rows})) + ")"
        else:
            basis = f"{len(rows)} shop quote(s): " + ", ".join(sorted({r.get('shop', '') for r in rows}))
        return {"min": lo, "max": max(lo, hi), "basis": basis, "is_estimate": indicative}

def add_quote(path, part, severity, shop, lo, hi, note=""):
    path = Path(path)
    new = not path.is_file()
    needs_newline = False
    if not new:
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                needs_newline = f.read(1) not in (b"\n", b"\r")
    with open(path, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        w = csv.writer(f)
        if new:
            w.writerow(RepairQuotes.FIELDS)
        w.writerow([part, severity, shop, lo, hi, "shop quote", note])


# ================================================================ live part prices
def _parse_year_spans(title):
    spans = []
    for m in re.finditer(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(\d{2,4})\b", title, flags=re.I):
        lo, end = int(m.group(1)), m.group(2)
        if len(end) == 4:
            hi = int(end)
        elif len(end) == 2:
            hi = int(str(lo)[:2] + end)
        else:
            continue
        if lo <= hi <= 2100:
            spans.append((lo, hi))
    return spans


def _fits_year(title, year):
    """True / False if the title states a year range, None if it states none."""
    spans = _parse_year_spans(title)
    if not spans or not year:
        return None
    return any(lo <= year <= hi for lo, hi in spans)

def _stats(prices):
    prices = sorted(prices)
    if len(prices) >= 4:
        q = statistics.quantiles(prices, n=4)
        low, high = q[0], q[2]
    else:
        low, high = prices[0], prices[-1]
    return {"n": len(prices), "low": int(low), "median": int(statistics.median(prices)), "high": int(high)}


class PriceFinder:
    def __init__(self):
        self.cache = json.loads(CACHE_FILE.read_text("utf-8")) if CACHE_FILE.is_file() else {}
        self.driver = None
        self.debug = False

    # ---------- browser
    
    def _get_driver(self):
        if self.driver is None:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            options = Options()
            options.page_load_strategy = "eager"
            options.add_argument("--headless=new")
            options.add_argument("--window-size=1280,900")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")

            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)
            self.driver.set_script_timeout(20)

        return self.driver


    def close(self):
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
    
    def _scrape(self, query):
        """Search PakWheels' accessories page and read titles/prices from result cards.

        Selectors grounded in the user's Inspect snippets:
          search field: input#search_key[name="search_key"]
          title:        h3
          price:        div.price-details

        The supplied HTML did not include the enclosing result-card element, so
        find each h3's nearest ancestor that contains a price element.
        """

        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        d = self._get_driver()

        try:
            d.get(BASE_URL)
        except TimeoutException:
        # Stop loading if the page's HTML is already available.
            try:
                d.execute_script("window.stop();")
            except WebDriverException:
                print("[PakWheels: unable to stop timed-out page load]")
                return []
        except WebDriverException as exc:
            print(f"[PakWheels: page navigation failed: {exc}]")
            return []
    
        wait = WebDriverWait(d, 15)

        try:
            search = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "input#search_key[name='search_key']")
                )
        )
            search.clear()
            search.send_keys(query)
            search.send_keys(Keys.ENTER)

            wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.price-details")
            )
        )

        except TimeoutException:
            print(f"[PakWheels: search/results timed out for {query!r}]")
            return []


        out, seen = [], set()
        for heading in d.find_elements(By.CSS_SELECTOR, TITLE_CSS):
            try:
                title = (heading.text or heading.get_attribute("textContent") or "").strip()
                if not title:
                    continue

                # Walk upward to the closest element containing a price. The
                # exact card wrapper class was not present in the supplied HTML.
                card = d.execute_script(
                    """
                    let node = arguments[0];
                    while (node && node !== document.body) {
                      if (node.querySelector && node.querySelector('div.price-details')) return node;
                      node = node.parentElement;
                    }
                    return null;
                    """,
                    heading,
                )
                if card is None:
                    continue

                price_elements = card.find_elements(By.CSS_SELECTOR, PRICE_CSS)
                if not price_elements:
                    continue
                price_text = (price_elements[0].text or price_elements[0].get_attribute("textContent") or "").strip()
                match = re.search(r"(?:PKR|Rs\.?|₨)\s*([\d,]+)", price_text, flags=re.I)
                if not match:
                    # Some page versions may expose only the numeric text.
                    match = re.search(r"\b([\d,]{3,})\b", price_text)
                if not match:
                    continue
                price = int(match.group(1).replace(",", ""))
                if price <= 0:
                    continue
                links = card.find_elements(By.CSS_SELECTOR, "a[href]")
                link = links[0].get_attribute("href") if links else ""
                dedupe_key = link or f"{title.lower()}|{price}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                cats = [e.text.strip() for e in card.find_elements(By.CSS_SELECTOR, ".search-vehicle-info li") if e.text.strip()]
                out.append({"title": title, "price": price, "url": link, "cats": cats})
            except Exception as exc:
                if getattr(self, "debug", False):
                    print(f"[PakWheels listing skipped: {exc}]")
                continue

        time.sleep(DELAY_SECONDS)
        return out

    # ---------- price lookup
    def get_prices(self, make, model, year, part, debug=False):
        self.debug = debug
        cfg = PARTS.get(part)
        if cfg is None:
            return None
        year = int(year) if str(year).isdigit() else None
        key = f"pakwheels-v2|{make}|{model}|{year}|{part}".lower()
        hit = self.cache.get(key)
        if hit and time.time() - hit["ts"] < CACHE_DAYS * 86400 and not debug:
            return hit["stats"]

        listings = []
        for term in cfg["terms"]:
            listings += self._scrape(f"{make} {model} {term}".strip())

        kept, dropped = [], []
        for l in listings:
            t = l["title"].lower() + " " + " ".join(l["cats"]).lower()
            reason = None
            if not any(w in t for w in cfg["must"]):
                reason = "missing keyword"
            elif any(w in t for w in cfg["block"]):
                reason = "blocked word"
            elif l["price"] < cfg["min"]:
                reason = f"price < {cfg['min']}"
            (dropped if reason else kept).append({**l, "reason": reason})

        # prefer listings whose stated year range contains the car's year
        tiers = {True: [], None: [], False: []}
        for l in kept:
            tiers[_fits_year(l["title"], year)].append(l)
        chosen, year_match = tiers[True], "exact"
        if len(chosen) < 2:
            chosen, year_match = tiers[True] + tiers[None], "unknown-year included"
        if not chosen:
            chosen, year_match = kept, "year not matched"

        if debug:
            print(f"\n=== {part}: {len(listings)} listings, {len(kept)} kept, {len(dropped)} dropped, using '{year_match}'")
            for l in dropped:
                print(f"  DROP ({l['reason']}): PKR {l['price']:>7,}  {l['title'][:80]}")
            for l in chosen:
                print(f"  USE:             PKR {l['price']:>7,}  {l['title'][:80]}")

        if not chosen:
            stats = None
        else:
            stats = _stats([l["price"] for l in chosen])
            stats["year_match"] = year_match
            stats["examples"] = [{"title": l["title"], "price": l["price"], "url": l["url"]} for l in chosen[:3]]
        self.cache[key] = {"ts": time.time(), "stats": stats}
        CACHE_FILE.write_text(json.dumps(self.cache, indent=1), "utf-8")
        return stats
    def replace_cost(self, make, model, year, part, include_labour=False):
        """Part price range from live listings. Labour is excluded unless include_labour=True."""
        stats = self.get_prices(make, model, year, part, debug=getattr(self, "debug", False))
        if not stats:
            return None
        group = "body" if part in BODY_PANELS else "other"
        extra = (FITTING_PKR[group] + PAINT_PKR[group]) if include_labour else 0
        return {"min": stats["low"] + extra, "max": stats["high"] + extra, "is_estimate": True,
                "basis": {"source": "PakWheels live listings", "labour_included": include_labour,
                          "part_price_low": stats["low"], "part_price_median": stats["median"],
                          "part_price_high": stats["high"], "listings_used": stats["n"],
                          "year_match": stats["year_match"], "examples": stats["examples"]}}


# ================================================================ repair AND replace for one part
class CostEstimator:
    """cost = CostEstimator(live=True).both({"make": "Toyota", "model": "Corolla", "year": "2018"}, "hood", "moderate")
       -> {"recommended": "repair", "repair": {...} or None, "replace": {...} or None}
       All figures are PKR per single part."""

    def __init__(self, live=False, include_labour=False, quotes=None):
        self.include_labour = include_labour
        self.quotes = RepairQuotes() if quotes is None else quotes
        self.finder = PriceFinder() if live else None

    def close(self):
        if self.finder is not None:
            self.finder.close()

    def both(self, vehicle, part, severity):
        repair = self._repair(part, severity)
        replace = self._replace(vehicle, part)
        return {"recommended": self._recommend(part, severity, repair, replace),
                "repair": repair, "replace": replace}

    def _repair(self, part, severity):
        if part in ALWAYS_REPLACE or severity not in ("minor", "moderate"):
            return None
        quote = self.quotes.lookup(part, severity)
        if quote:
            return quote
        # Public indicative repair figures are still from repair_quotes.csv,
        # not replacement prices and not a generated CSV fallback.
        return self.quotes.lookup(part, severity, indicative=True)

    def _replace(self, vehicle, part):
        if self.finder is None:
            return None
        make = vehicle.get("make", "")
        model = vehicle.get("model", "")
        year = vehicle.get("year", "")
        if not make or not model:
            return None
        try:
            return self.finder.replace_cost(make, model, year, part, self.include_labour)
        except Exception as ex:
            print(f"[PakWheels live price lookup failed for {part}: {ex}]")
            return None

    @staticmethod
    def _recommend(part, severity, repair, replace):
        if part in ALWAYS_REPLACE or severity == "severe":
            return "replace" if replace else ("repair" if repair else None)
        if repair and replace and repair["min"] > replace["max"]:
            return "replace"
        return "repair" if repair else ("replace" if replace else None)


# ================================================================ command line
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-quote", action="store_true", help="append a shop quote to repair_quotes.csv")
    ap.add_argument("--make")
    ap.add_argument("--model")
    ap.add_argument("--year", default="")
    ap.add_argument("--part", choices=sorted(set(PARTS) | {"license_plate"}))
    ap.add_argument("--severity", default="moderate", choices=["minor", "moderate", "severe"])
    ap.add_argument("--shop", help="shop name (for --add-quote)")
    ap.add_argument("--min", type=float, dest="lo", help="lowest quote in PKR (for --add-quote)")
    ap.add_argument("--max", type=float, dest="hi", help="highest quote in PKR (for --add-quote)")
    ap.add_argument("--note", default="")
    ap.add_argument("--debug", action="store_true", help="print kept/dropped listings (ignores cache)")
    a = ap.parse_args()

    if a.add_quote:
        if not (a.part and a.shop and a.lo is not None and a.hi is not None):
            ap.error("--add-quote needs --part, --severity, --shop, --min and --max")
        add_quote(REPAIR_QUOTES_FILE, a.part, a.severity, a.shop, a.lo, a.hi, a.note)
        print(f"Saved quote to {REPAIR_QUOTES_FILE.name}: {a.shop} / {a.part} / {a.severity}: PKR {a.lo:,.0f} - {a.hi:,.0f}")
    else:
        if not (a.make and a.model and a.part):
            ap.error("--make, --model and --part are required")
        est = CostEstimator(live=True)
        try:
            if a.debug and est.finder is not None:
                est.finder.debug = True

            vehicle = {"make": a.make, "model": a.model, "year": a.year}
            print(json.dumps(est.both(vehicle, a.part, a.severity), indent=2))
        finally:
            est.close()