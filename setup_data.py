"""One-time setup on a new computer: fetch the public datasets that are not stored in this repository.

  pip install -r requirements.txt
  python setup_data.py

data_raw/greco is required (UFCStats mirror, updated daily). The others are archives used by research scripts
(UFC_Final holds the 2008-2020 Pinnacle closing odds). Page caches rebuild themselves on the next update.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOS = {
    "greco": "https://github.com/Greco1899/scrape_ufc_stats.git",
    "UFC_Final": "https://github.com/iankotliar/UFC_Final.git",
    "ultimate_ufc_dataset": "https://github.com/shortlikeafox/ultimate_ufc_dataset.git",
    "ufc-data": "https://github.com/jansen88/ufc-data.git",
}

if __name__ == "__main__":
    for name, url in REPOS.items():
        dest = ROOT / "data_raw" / name
        if dest.exists():
            print("already there:", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    print("done. Double-click-app shortcut: create one pointing to pythonw.exe -m ufc.app in this folder (see README).")
