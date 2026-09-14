"""Polite HTTP fetcher with an on-disk cache so scrapes are resumable and re-runs are incremental."""
import gzip, hashlib, time, threading
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_raw" / "cache"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
# Wikimedia asks API clients to identify themselves with a descriptive agent
_WM = {"User-Agent": "UFCPredictorResearch/0.1 (personal non-commercial research script; python-requests)"}
HOST_HEADERS = {"en.wikipedia.org": _WM, "wikimedia.org": _WM}
_last = {}
_lock = threading.Lock()


def _path(url):
    host = url.split("/")[2].replace("www.", "")
    return CACHE / host / (hashlib.sha1(url.encode()).hexdigest() + ".html.gz")


def fetch(url, max_age_days=None, delay=1.0, retries=4, min_len=500):
    """Return page text. Cached copies are reused unless older than max_age_days (None = forever)."""
    p = _path(url)
    if p.exists() and (max_age_days is None or time.time() - p.stat().st_mtime < max_age_days * 86400):
        return gzip.decompress(p.read_bytes()).decode("utf-8")
    host = url.split("/")[2]
    for attempt in range(retries):
        with _lock:  # one request per host every `delay` seconds
            wait = _last.get(host, 0) + delay - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[host] = time.time()
        try:
            r = requests.get(url, headers=HOST_HEADERS.get(host, UA), timeout=40)
            if r.status_code == 404:
                return None
            if r.status_code == 429:  # rate limited: back off hard
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 200 and len(r.text) >= min_len:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(gzip.compress(r.text.encode("utf-8")))
                return r.text
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")
