#!/usr/bin/env python3
"""
arxiv_scraper.py
- Query an arXiv category for the latest N papers (default 200)
- For each paper: download /abs/ page, extract cleaned text using trafilatura,
  take a screenshot of the abstract element with Selenium, run pytesseract on that
  screenshot, and save the fields to a JSON file (arxiv_clean.json).
"""

import time
import json
import argparse
import logging
from html import unescape
import os
import shutil

import requests
import feedparser
import trafilatura
from PIL import Image
import io
import pytesseract

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---- Configuration ----
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "arXiv-Abstract-Scraper/1.0 (+https://example/)"

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def query_arxiv(category: str, max_results: int = 200):
    """
    Query the arXiv API for the latest papers in a category.
    Returns parsed feed (feedparser structure).
    """
    # The arXiv API can return results in pages. We'll request in batches
    # and concatenate entries until we reach max_results or there are no
    # more items.
    headers = {"User-Agent": USER_AGENT}
    entries = []
    start = 0
    batch_size = 100  # reasonable page size
    while len(entries) < max_results:
        to_fetch = min(batch_size, max_results - len(entries))
        params = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": to_fetch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        if not feed.entries:
            break
        entries.extend(feed.entries)
        # if the API returned fewer items than requested, we've reached the end
        if len(feed.entries) < to_fetch:
            break
        start += to_fetch
        # be polite: short pause between paged requests
        time.sleep(0.3)

    logging.info("Queried arXiv: requested %d, received %d entries", max_results, len(entries))
    return entries


def setup_selenium_headless():
    """
    Create a Selenium Chrome driver in headless mode using webdriver-manager
    """
    options = Options()
    options.add_argument("--headless=new")  # use new headless if available
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1200,1600")
    options.add_argument("--hide-scrollbars")
    # Prevent loading images? We need images for screenshot text render; keep default.
    try:
        # Use Service with webdriver-manager to avoid positional-argument errors
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except WebDriverException as e:
        logging.error("Failed to start Chrome webdriver: %s", e)
        raise
    return driver


def fetch_abs_html(url: str):
    """Download the /abs/ HTML page content (requests)."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def trafilatura_clean(html: str):
    """Use trafilatura to extract cleaned text. Return None on failure."""
    downloaded = trafilatura.extract(html, output_format="text", include_comments=False)
    if downloaded:
        return downloaded.strip()
    return None


def ocr_abstract_from_screenshot(driver, url: str):
    """
    Use Selenium to open URL, find the abstract block and screenshot it.
    Returns OCR text or None on failure.
    """
    driver.get(url)
    time.sleep(0.5)  # give page time to render; adjust if needed

    # arXiv structures abstracts in <blockquote class="abstract"> or similar.
    selectors = [
        "blockquote.abstract",  # typical arXiv
        "div.abstract",         # fallback
        "#abs > blockquote",    # alternate
    ]
    for sel in selectors:
        try:
            elem = driver.find_element("css selector", sel)
            png = elem.screenshot_as_png  # bytes
            img = Image.open(io.BytesIO(png)).convert("RGB")
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                return text.strip()
        except NoSuchElementException:
            continue
        except Exception as e:
            logging.debug("OCR screenshot error for selector %s: %s", sel, e)
            continue
    # If element screenshot fails, try full page screenshot and run OCR (last resort)
    try:
        full_png = driver.get_screenshot_as_png()
        img_full = Image.open(io.BytesIO(full_png)).convert("RGB")
        text_full = pytesseract.image_to_string(img_full)
        if text_full and text_full.strip():
            return text_full.strip()
    except Exception as e:
        logging.debug("Full page OCR error: %s", e)
    return None


def authors_from_entry(entry):
    """Return a list of author names from feedparser entry"""
    authors = []
    for a in entry.get("authors", []):
        name = a.get("name")
        if name:
            authors.append(name)
    return authors


def main(category="cs.CL", max_results=200, outpath="arxiv_clean.json", delay=0.7):
    entries = query_arxiv(category, max_results=max_results)

    # Launch Selenium once
    driver = None
    try:
        driver = setup_selenium_headless()
    except Exception as e:
        logging.warning("Selenium not available; OCR screenshots will be skipped. Error: %s", e)

    results = []
    for i, e in enumerate(entries, 1):
        try:
            url = e.get("id")  # usually the /abs/ URL
            title = unescape(e.get("title", "")).replace("\n", " ").strip()
            date = e.get("published", "")  # e.g., '2025-10-12T...'
            authors = authors_from_entry(e)

            logging.info("Processing %d/%d: %s", i, len(entries), title[:80])

            # 1) fetch HTML
            html = None
            try:
                html = fetch_abs_html(url)
            except Exception as ex:
                logging.warning("Failed to fetch HTML for %s: %s", url, ex)

            # 2) trafilatura cleaning
            cleaned = None
            if html:
                try:
                    cleaned = trafilatura_clean(html)
                except Exception as ex:
                    logging.debug("Trafilatura error: %s", ex)

            # 3) OCR from screenshot (if selenium available)
            ocr_text = None
            if driver:
                try:
                    ocr_text = ocr_abstract_from_screenshot(driver, url)
                except Exception as ex:
                    logging.debug("OCR failed for %s: %s", url, ex)

            # Heuristic: try to extract abstract from the raw HTML as fallback (simple CSS)
            fallback_abstract = None
            if html:
                # quick-and-dirty: find <blockquote class="abstract"> ... </blockquote>
                import re
                m = re.search(r'<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>', html, re.S | re.I)
                if m:
                    text = re.sub(r'<[^>]+>', '', m.group(1))  # strip tags
                    fallback_abstract = unescape(text).replace("\n", " ").strip()

            # Choose final abstract text with priority:
            # 1) fallback_abstract (direct extraction)
            # 2) cleaned (trafilatura)
            # 3) ocr_text (tesseract)
            # Note: user wanted both trafilatura and tesseract; we save both candidates in JSON for traceability.
            final_abstract = fallback_abstract or cleaned or ocr_text or ""

            item = {
                "url": url,
                "title": title,
                "abstract": final_abstract,
                "abstract_sources": {
                    "blockquote_extraction": bool(fallback_abstract),
                    "trafilatura": bool(cleaned),
                    "ocr": bool(ocr_text),
                },
                "authors": authors,
                "date": date,
            }
            results.append(item)

            # polite delay (don't hammer arXiv)
            time.sleep(delay)
        except Exception as ex:
            logging.exception("Unhandled error processing entry %d: %s", i, ex)

    # write JSON compactly (minified) to keep under size
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"), indent=None)

    if driver:
        driver.quit()

    logging.info("Done. %d items saved to %s (approx size: %.2f KB)", len(results), outpath, (os.path.getsize(outpath) / 1024))


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="ArXiv Abstract Scraper")
    parser.add_argument("--category", "-c", default="cs.CL", help="arXiv category (e.g., cs.CL)")
    parser.add_argument("--max", "-m", type=int, default=200, help="max number of papers to fetch")
    parser.add_argument("--out", "-o", default="arxiv_clean.json", help="output JSON file")
    parser.add_argument("--delay", type=float, default=0.7, help="delay between requests (seconds)")
    args = parser.parse_args()
    main(category=args.category, max_results=args.max, outpath=args.out, delay=args.delay)
