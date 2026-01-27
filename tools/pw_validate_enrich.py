#!/usr/bin/env python3
"""
pw_validate_enrich.py

Purpose:
- Validate an existing catalog JSON (catalog-v1.json)
- Re-scrape ONLY records that are missing fields Plot depends on
- Output a new catalog file + reports (never overwrite the original)

Factual-only rules:
- Only fill fields if explicitly present on the plant's source_url page.
- If a value can't be found on the page, it stays blank/None.
- Always preserve source_url + source_date; add source_checked date.

Usage (from repo root on Windows):
  pip install requests beautifulsoup4
  python tools\\pw_validate_enrich.py --in catalog\\catalog-v1.json --out catalog\\catalog-v1.1.json

Optional:
  --delay 0.25         delay between requests (seconds)
  --max 0              0 = no limit; otherwise only process N plants (for testing)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

UA = "PlotCatalogValidator/1.0 (factual-only; respectful crawl)"

SUN_ORDER = ["full_sun", "part_sun", "part_shade", "shade"]

# ---------- helpers ----------

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

def season_bin(s: str) -> str:
    s = clean(s).lower()
    if "early spring" in s: return "early_spring"
    if "mid spring" in s or s == "spring": return "mid_spring"
    if "late spring" in s: return "late_spring"
    if "early summer" in s: return "early_summer"
    if "mid summer" in s or s == "summer": return "mid_summer"
    if "late summer" in s: return "late_summer"
    if "fall" in s or "autumn" in s: return "fall"
    if "winter" in s: return "winter"
    return ""

def map_light(light: str) -> Optional[str]:
    l = clean(light).lower()
    if l in ("sun", "full sun"): return "full_sun"
    if "part sun" in l: return "part_sun"
    if "part shade" in l: return "part_shade"
    if "shade" in l: return "shade"
    return None

def map_water(water: str) -> List[str]:
    w = clean(water).lower()
    if "low" in w: return ["dry"]
    if "average" in w or "medium" in w: return ["average"]
    if "moist" in w: return ["moist"]
    if "high" in w or "wet" in w: return ["wet"]
    return []

def parse_zones(zone_text: str) -> Tuple[Optional[int], Optional[int]]:
    nums = [int(n) for n in re.findall(r"\b([0-9]{1,2})\s*[ab]?\b", zone_text)]
    if not nums:
        return None, None
    return min(nums), max(nums)

def parse_ft_range(text: str, label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Handles:
      Height: 3' - 5'
      Height: 3'
    """
    apost = r"[’']"
    m = re.search(rf"{label}:\s*([0-9.]+){apost}\s*-\s*([0-9.]+){apost}", text, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"{label}:\s*([0-9.]+){apost}", text, re.I)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None

def parse_in_range(text: str, label: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Handles inch-based entries (rare but possible):
      Height: 18" - 24"
      Height: 18"
    Returns feet.
    """
    m = re.search(rf"{label}:\s*([0-9.]+)\"\s*-\s*([0-9.]+)\"", text, re.I)
    if m:
        return float(m.group(1))/12.0, float(m.group(2))/12.0
    m = re.search(rf"{label}:\s*([0-9.]+)\"", text, re.I)
    if m:
        v = float(m.group(1))/12.0
        return v, v
    return None, None

def best_name_fields(soup: BeautifulSoup, full_text: str) -> Tuple[str, str, str]:
    """
    Prefer title parsing, but fallback to visible H1/H2 lines if needed.
    Returns (trade_name, common, botanical).
    """
    trade = common = botanical = ""
    title = soup.title.get_text(strip=True) if soup.title else ""
    left = title.split("|")[0].strip()
    parts = [p.strip() for p in left.split(" - ") if p.strip()]
    if len(parts) >= 1: trade = parts[0]
    if len(parts) >= 2: common = parts[1]
    if len(parts) >= 3: botanical = parts[2]

    # Fallback: find a likely header chunk
    if not (trade or common or botanical):
        for h in soup.find_all(["h1", "h2"]):
            t = clean(h.get_text(" ", strip=True))
            if t and len(t) < 120:
                # best-effort: keep as trade if nothing else
                trade = trade or t

    # Botanical fallback: look for a latin-looking line in the text
    if not botanical:
        # common PW pages sometimes include botanical in title only; if missing, leave blank
        pass

    return trade, common, botanical

def extract_field(full: str, label: str) -> str:
    m = re.search(rf"{re.escape(label)}\s*:\s*(.+)", full, re.I)
    return clean(m.group(1)) if m else ""

def scrape_pw_page(url: str) -> Dict[str, Any]:
    """
    Scrape only factual fields from PW plant page text.
    Returns dict of possible fields; caller decides what to apply.
    """
    html = http_get(url)
    soup = BeautifulSoup(html, "html.parser")
    strip_scripts(soup)

    full = "\n".join([clean(x) for x in soup.get_text("\n").split("\n") if clean(x)])

    trade, common, botanical = best_name_fields(soup, full)

    # Zones
    zmin = zmax = None
    ztxt = extract_field(full, "Hardiness Zones")
    if ztxt:
        zmin, zmax = parse_zones(ztxt)

    # Sun
    sun_set = set()
    for l in re.findall(r"Light Requirement:\s*([A-Za-z ]+)", full, re.I):
        mapped = map_light(l)
        if mapped:
            sun_set.add(mapped)
    sun = sorted(sun_set, key=lambda x: SUN_ORDER.index(x) if x in SUN_ORDER else 99)

    # Moisture
    water = extract_field(full, "Water Category")
    moisture = map_water(water) if water else []

    # Size (feet max)
    hmin, hmax = parse_ft_range(full, "Height")
    wmin, wmax = parse_ft_range(full, "Spread")
    if hmax is None:
        _, hmax = parse_in_range(full, "Height")
    if wmax is None:
        _, wmax = parse_in_range(full, "Spread")

    # Bloom (season bins)
    blooms = [season_bin(b) for b in re.findall(r"Bloom Time:\s*([A-Za-z ]+)", full, re.I)]
    blooms = [b for b in blooms if b]
    bloom_start = blooms[0] if blooms else ""
    bloom_end = blooms[-1] if blooms else ""

    # Type (if available)
    ptype = extract_field(full, "Plant Type").lower()

    # Conservative tags (only exact phrase presence)
    lt = full.lower()
    tags = set()
    if "drought tolerant" in lt: tags.add("drought_tolerant")
    if "fragrant" in lt: tags.add("fragrant")
    if "fall interest" in lt or "fall color" in lt: tags.add("fall_color")
    if "butterflies" in lt or "bees" in lt or "hummingbirds" in lt: tags.add("pollinator")

    return {
        "trade_name": trade,
        "common": common,
        "botanical": botanical,
        "zone_min": zmin,
        "zone_max": zmax,
        "sun": sun,
        "moisture": moisture,
        "height_ft_max": hmax,
        "width_ft_max": wmax,
        "bloom_start": bloom_start,
        "bloom_end": bloom_end,
        "type": ptype,
        "tags": sorted(tags),
        "source_checked": date.today().isoformat(),
    }

# ---------- validation ----------

BLOOM_ORDER = ["early_spring","mid_spring","late_spring","early_summer","mid_summer","late_summer","fall","winter"]

def is_valid_bloom(x: str) -> bool:
    return (x in BLOOM_ORDER) or (x == "")

def validate_record(p: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    if not p.get("id"):
        issues.append("missing:id")
    if not p.get("source_url"):
        issues.append("missing:source_url")
    if not (p.get("common") or p.get("trade_name") or p.get("botanical")):
        issues.append("missing:name_any")
    if p.get("zone_min") is None or p.get("zone_max") is None:
        issues.append("missing:zones")
    if not p.get("sun"):
        issues.append("missing:sun")
    if p.get("height_ft_max") in (None, ""):
        issues.append("missing:height_ft_max")
    if p.get("width_ft_max") in (None, ""):
        issues.append("missing:width_ft_max")
    if not is_valid_bloom(p.get("bloom_start","")):
        issues.append("invalid:bloom_start")
    if not is_valid_bloom(p.get("bloom_end","")):
        issues.append("invalid:bloom_end")
    # bloom missing is allowed, but flagged because your Seasons needs it
    if not p.get("bloom_start"):
        issues.append("missing:bloom_start")
    if not p.get("bloom_end"):
        issues.append("missing:bloom_end")
    if not p.get("moisture"):
        issues.append("missing:moisture")

    return issues

def should_rescrape(issues: List[str]) -> bool:
    # Only rescrape if missing fields that the PW page might contain
    wanted_prefixes = (
        "missing:height_ft_max",
        "missing:width_ft_max",
        "missing:bloom_start",
        "missing:bloom_end",
        "missing:moisture",
        "missing:sun",
        "missing:zones",
        "missing:name_any",
    )
    return any(i.startswith(wanted_prefixes) for i in issues)

def apply_factual_updates(p: Dict[str, Any], scraped: Dict[str, Any]) -> int:
    """
    Only fill blanks / Nones. Never overwrite a value already present.
    Returns number of fields updated.
    """
    updated = 0

    def fill(key: str):
        nonlocal updated
        if key not in scraped:
            return
        cur = p.get(key)
        new = scraped.get(key)
        # Consider empty list as missing for list fields
        missing = (cur is None) or (cur == "") or (cur == []) or (cur == {})
        if missing and (new not in (None, "", [], {})):
            p[key] = new
            updated += 1

    for k in ["common","botanical","trade_name","zone_min","zone_max","sun","moisture","height_ft_max","width_ft_max","bloom_start","bloom_end","type"]:
        fill(k)

    # Tags: if tags empty, adopt scraped tags (still factual)
    if (p.get("tags") in (None, [], "")) and scraped.get("tags"):
        p["tags"] = scraped["tags"]
        updated += 1

    # Always add source_checked (auditing) even if no other updates
    p["source_checked"] = scraped.get("source_checked", date.today().isoformat())

    return updated

# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input catalog json (e.g., catalog\\catalog-v1.json)")
    ap.add_argument("--out", dest="out", required=True, help="output catalog json (e.g., catalog\\catalog-v1.1.json)")
    ap.add_argument("--delay", type=float, default=0.25, help="delay between requests (seconds)")
    ap.add_argument("--max", type=int, default=0, help="limit processing to N plants (0=no limit)")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    plants: List[Dict[str, Any]] = data.get("plants", [])
    if not isinstance(plants, list):
        raise SystemExit("Input JSON does not contain a 'plants' array.")

    max_n = args.max if args.max and args.max > 0 else len(plants)

    # Reports folder
    reports_dir = "reports"
    try:
        import os
        os.makedirs(reports_dir, exist_ok=True)
    except Exception:
        pass

    # Initial validation
    initial_issues = []
    for p in plants[:max_n]:
        issues = validate_record(p)
        if issues:
            initial_issues.append((p.get("id",""), p.get("source_url",""), ";".join(issues)))

    # Targeted re-scrape
    total_updates = 0
    rescraped = 0
    for idx, p in enumerate(plants[:max_n], start=1):
        issues = validate_record(p)
        if not issues or not should_rescrape(issues):
            continue

        url = p.get("source_url","")
        if not url:
            continue

        try:
            scraped = scrape_pw_page(url)
        except Exception as e:
            # Log scrape failure but do not crash the run
            with open(f"{reports_dir}/scrape_failures.csv", "a", newline="", encoding="utf-8") as wf:
                w = csv.writer(wf)
                w.writerow([p.get("id",""), url, str(e)])
            time.sleep(args.delay)
            continue

        changed = apply_factual_updates(p, scraped)
        total_updates += changed
        rescraped += 1

        if idx % 50 == 0:
            print(f"[progress] {idx}/{max_n} checked | rescraped={rescraped} total_updates={total_updates}")

        time.sleep(args.delay)

    # Final validation
    final_issues = []
    for p in plants[:max_n]:
        issues = validate_record(p)
        if issues:
            final_issues.append((p.get("id",""), p.get("source_url",""), ";".join(issues)))

    # Write updated catalog (new file)
    out_data = dict(data)
    out_data["updated"] = date.today().isoformat()
    out_data["version"] = str(out_data.get("version","")) + "+validated"
    out_data["plants"] = plants

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # Write reports
    with open(f"{reports_dir}/validation_before.csv", "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(["id","source_url","issues"])
        for row in initial_issues:
            w.writerow(row)

    with open(f"{reports_dir}/validation_after.csv", "w", newline="", encoding="utf-8") as wf:
        w = csv.writer(wf)
        w.writerow(["id","source_url","issues"])
        for row in final_issues:
            w.writerow(row)

    summary = {
        "date": date.today().isoformat(),
        "plants_total": len(plants),
        "plants_checked": max_n,
        "initial_problem_records": len(initial_issues),
        "final_problem_records": len(final_issues),
        "rescraped_records": rescraped,
        "fields_filled_total": total_updates,
        "notes": "Factual-only: fields filled only when explicitly found on source_url pages."
    }
    with open(f"{reports_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[done] wrote:", args.out)
    print("[done] reports:", f"{reports_dir}/validation_before.csv", f"{reports_dir}/validation_after.csv", f"{reports_dir}/summary.json")
    print("[summary]", summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
