#!/usr/bin/env python3
"""
pw_build_catalog.py

Build a Plot-ready catalog JSON by crawling official Proven Winners PROGRAM pages,
then scraping each plant page for key attributes and filtering to Zone 5/6.

Hard requirement: factual-only extraction.
- No guessing/inference for fields not explicitly found on the page.
- Every record includes source_url + source_date for auditing.

Output:
  catalog/catalog-v1.json (default path you pass via --out)

Usage (from repo root on Windows):
  pip install requests beautifulsoup4
  python tools\pw_build_catalog.py --zones 5 6 --out catalog\catalog-v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup


BASE = "https://www.provenwinners.com"
UA = "PlotCatalogBuilder/1.0 (factual-only extraction; respectful crawl)"


# Official Proven Winners program catalog pages (edit here if you want more/less).
PROGRAMS = [
    {"name": "Proven Winners Shrubs", "url": f"{BASE}/plants/program/proven-winners-shrubs", "series": "Proven Winners"},
    {"name": "Proven Winners Perennials", "url": f"{BASE}/plants/program/proven-winners-perennials", "series": "Proven Winners"},
    {"name": "Proven Winners Annuals", "url": f"{BASE}/plants/program/proven-winners-annuals", "series": "Proven Winners"},
]

SUN_ORDER = ["full_sun", "part_sun", "part_shade", "shade"]


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def http_get(url: str, timeout_s: int = 30) -> str:
    r = requests.get(
        url,
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.text


def strip_scripts(soup: BeautifulSoup) -> None:
    for t in soup(["script", "style", "noscript"]):
        t.decompose()


def is_plant_page(abs_url: str) -> bool:
    # Plant pages usually: /plants/<plant-group>/<slug>
    p = urlparse(abs_url)
    parts = p.path.strip("/").split("/")
    if len(parts) < 3:
        return False
    if parts[0] != "plants":
        return False
    if parts[1] in ("program", "search"):
        return False
    return True


def extract_plant_links(listing_html: str, listing_url: str) -> Set[str]:
    soup = BeautifulSoup(listing_html, "html.parser")
    urls: Set[str] = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(listing_url, a["href"])
        if is_plant_page(u):
            urls.add(u.split("#")[0])
    return urls


def parse_showing_total(text: str) -> Optional[int]:
    m = re.search(r"Showing\s+\d+\s*-\s*\d+\s+of\s+(\d+)", text, re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def find_page2_url(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    # Find an anchor with text "2" in pagination
    for a in soup.find_all("a", href=True):
        if a.get_text(strip=True) == "2":
            return urljoin(base_url, a["href"])
    return None


def build_paginated_urls(program_url: str) -> List[str]:
    """
    PW program pages are paginated. We:
      - fetch first page
      - read "Showing 1 - 15 of N"
      - find the link to page "2" to infer ?page= style
      - generate URLs for all pages
    """
    first_html = http_get(program_url)
    soup = BeautifulSoup(first_html, "html.parser")
    strip_scripts(soup)
    text = soup.get_text("\n")

    total = parse_showing_total(text)
    if not total:
        return [program_url]

    per_page = 15
    pages = (total + per_page - 1) // per_page
    if pages <= 1:
        return [program_url]

    page2 = find_page2_url(soup, program_url)
    if not page2:
        return [program_url]

    u2 = urlparse(page2)
    q2 = parse_qs(u2.query)
    if "page" not in q2 or not q2["page"]:
        return [program_url]

    try:
        base_page_val = int(q2["page"][0])
    except Exception:
        return [program_url]

    # Build base URL with all params except page
    base_u = urlparse(program_url)
    base_q = parse_qs(base_u.query)
    base_q.pop("page", None)
    base_query = urlencode({k: v[0] for k, v in base_q.items()})
    base = urlunparse(base_u._replace(query=base_query))

    urls = [base]
    # page 2 corresponds to base_page_val, page 3 to base_page_val+1, etc.
    for page_num in range(2, pages + 1):
        page_val = base_page_val + (page_num - 2)
        uu = urlparse(base)
        qq = parse_qs(uu.query)
        qq["page"] = [str(page_val)]
        query = urlencode({k: v[0] for k, v in qq.items()})
        urls.append(urlunparse(uu._replace(query=query)))

    return urls


def parse_zones(zone_text: str) -> Tuple[Optional[int], Optional[int]]:
    # Accepts strings like "5a, 5b, 6a, 6b" or "4-9"
    nums = [int(n) for n in re.findall(r"\b([0-9]{1,2})\s*[ab]?\b", zone_text)]
    if not nums:
        return None, None
    return min(nums), max(nums)


def map_light(light: str) -> Optional[str]:
    l = clean(light).lower()
    if l in ("sun", "full sun"):
        return "full_sun"
    if "part sun" in l:
        return "part_sun"
    if "part shade" in l:
        return "part_shade"
    if "shade" in l:
        return "shade"
    return None


def map_water(water: str) -> List[str]:
    # PW commonly shows: Low / Average / High
    w = clean(water).lower()
    if "low" in w:
        return ["dry"]
    if "average" in w or "medium" in w:
        return ["average"]
    if "moist" in w:
        return ["moist"]
    if "high" in w or "wet" in w:
        return ["wet"]
    return []


def parse_ft_range(text: str, label: str) -> Tuple[Optional[float], Optional[float]]:
    # Height: 3' - 5' or Height: 3'
    apost = r"[’']"
    m = re.search(rf"{label}:\s*([0-9.]+){apost}\s*-\s*([0-9.]+){apost}", text, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"{label}:\s*([0-9.]+){apost}", text, re.I)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None


def season_bin(s: str) -> str:
    s = clean(s).lower()
    if "early spring" in s:
        return "early_spring"
    if "mid spring" in s or s == "spring":
        return "mid_spring"
    if "late spring" in s:
        return "late_spring"
    if "early summer" in s:
        return "early_summer"
    if "mid summer" in s or s == "summer":
        return "mid_summer"
    if "late summer" in s:
        return "late_summer"
    if "fall" in s or "autumn" in s:
        return "fall"
    if "winter" in s:
        return "winter"
    return ""


def zone_ok(zmin: Optional[int], zmax: Optional[int], targets: List[int]) -> bool:
    if zmin is None or zmax is None:
        return False
    return any(zmin <= z <= zmax for z in targets)


@dataclass
class Plant:
    # Minimal Plot-ready factual fields (no guesses)
    id: str
    common: str
    botanical: str
    cultivar: str
    trade_name: str
    series: str
    type: str
    sun: List[str]
    moisture: List[str]
    zone_min: Optional[int]
    zone_max: Optional[int]
    height_ft_max: Optional[float]
    width_ft_max: Optional[float]
    bloom_start: str
    bloom_end: str
    tags: List[str]
    source_url: str
    source_date: str


def parse_title_fields(soup: BeautifulSoup) -> Tuple[str, str, str]:
    # Common PW title format: "<Trade> - <Common> - <Botanical> | Proven Winners"
    title = soup.title.get_text(strip=True) if soup.title else ""
    left = title.split("|")[0].strip()
    parts = [p.strip() for p in left.split(" - ") if p.strip()]
    trade = parts[0] if len(parts) >= 1 else ""
    common = parts[1] if len(parts) >= 2 else ""
    botanical = parts[2] if len(parts) >= 3 else ""
    return trade, common, botanical


def parse_plant_page(url: str, series_label: str) -> Optional[Plant]:
    html = http_get(url)
    soup = BeautifulSoup(html, "html.parser")
    strip_scripts(soup)
    text = soup.get_text("\n")
    lines = [clean(l) for l in text.split("\n")]
    lines = [l for l in lines if l]
    full = "\n".join(lines)

    trade, common, botanical = parse_title_fields(soup)

    # Hardiness Zones (required to include in Zone 5/6 selection)
    mz = re.search(r"Hardiness Zones:\s*([0-9ab,\s-]+)", full, re.I)
    if not mz:
        return None
    zmin, zmax = parse_zones(mz.group(1))

    # Light Requirement(s)
    sun_set: Set[str] = set()
    for l in re.findall(r"Light Requirement:\s*([A-Za-z ]+)", full, re.I):
        mapped = map_light(l)
        if mapped:
            sun_set.add(mapped)
    sun = sorted(sun_set, key=lambda x: SUN_ORDER.index(x) if x in SUN_ORDER else 99)

    # Water Category
    water = ""
    mw = re.search(r"Water Category:\s*([A-Za-z /-]+)", full, re.I)
    if mw:
        water = mw.group(1)
    moisture = map_water(water)

    # Size (use max values only)
    hmin, hmax = parse_ft_range(full, "Height")
    wmin, wmax = parse_ft_range(full, "Spread")

    # Bloom Time bins
    blooms = [season_bin(b) for b in re.findall(r"Bloom Time:\s*([A-Za-z ]+)", full, re.I)]
    blooms = [b for b in blooms if b]
    bloom_start = blooms[0] if blooms else ""
    bloom_end = blooms[-1] if blooms else ""

    # Plant Type (factual text; normalize to lowercase)
    ptype = ""
    mt = re.search(r"Plant Type:\s*([A-Za-z ]+)", full, re.I)
    if mt:
        ptype = clean(mt.group(1)).lower()

    # Cultivar code is not consistently exposed in page text; keep blank (factual-only)
    cultivar = ""

    # Conservative tags: only when the exact phrase appears in page text
    tags: List[str] = []
    lt = full.lower()
    if "drought tolerant" in lt:
        tags.append("drought_tolerant")
    if "fragrant" in lt:
        tags.append("fragrant")
    if "fall interest" in lt or "fall color" in lt:
        tags.append("fall_color")
    if "butterflies" in lt or "bees" in lt or "hummingbirds" in lt:
        tags.append("pollinator")

    slug = urlparse(url).path.rstrip("/").split("/")[-1]

    return Plant(
        id=slug,
        common=common,
        botanical=botanical,
        cultivar=cultivar,
        trade_name=trade,
        series=series_label,
        type=ptype,
        sun=sun,
        moisture=moisture,
        zone_min=zmin,
        zone_max=zmax,
        height_ft_max=hmax,
        width_ft_max=wmax,
        bloom_start=bloom_start,
        bloom_end=bloom_end,
        tags=sorted(set(tags)),
        source_url=url,
        source_date=date.today().isoformat(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="+", type=int, default=[5, 6], help="zones to include (e.g., 5 6)")
    ap.add_argument("--delay", type=float, default=0.25, help="delay between requests (seconds)")
    ap.add_argument("--out", type=str, required=True, help="output json path (e.g., catalog\\catalog-v1.json)")
    args = ap.parse_args()

    targets = args.zones
    delay = float(args.delay)

    all_plant_urls: Dict[str, str] = {}  # url -> series label (keep last if dup)
    for prog in PROGRAMS:
        print(f"[program] {prog['name']}: {prog['url']}")
        page_urls = build_paginated_urls(prog["url"])
        print(f"  pages detected: {len(page_urls)}")

        for i, page_url in enumerate(page_urls, start=1):
            print(f"  fetching page {i}/{len(page_urls)}")
            html = http_get(page_url)
            links = extract_plant_links(html, page_url)
            for u in links:
                all_plant_urls[u] = prog["series"]
            time.sleep(delay)

    urls = sorted(all_plant_urls.keys())
    print(f"[crawl] plant pages discovered: {len(urls)}")

    plants: List[Dict] = []
    kept = 0
    skipped = 0

    for idx, u in enumerate(urls, start=1):
        try:
            rec = parse_plant_page(u, all_plant_urls[u])
        except Exception as e:
            skipped += 1
            print(f"  [skip] error parsing: {u} :: {e}")
            time.sleep(delay)
            continue

        if rec and zone_ok(rec.zone_min, rec.zone_max, targets):
            plants.append(asdict(rec))
            kept += 1
        else:
            skipped += 1

        if idx % 50 == 0:
            print(f"  progress {idx}/{len(urls)} | kept={kept} skipped={skipped}")

        time.sleep(delay)

    payload = {
        "version": f"pw-zone-{'-'.join(str(z) for z in targets)}",
        "updated": date.today().isoformat(),
        "programs": [p["url"] for p in PROGRAMS],
        "plants": plants,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {len(plants)} plants to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
