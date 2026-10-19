"""Download only the two explicitly listed public HDB datasets, with receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import subprocess
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "resale": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "property": "d_17f5382f26140b1fdae0ba2ef6239d2f",
}


def get_bytes(url):
    # curl uses the host's working TLS/network stack; urllib receives HTTP 403 here.
    result = subprocess.run(["curl", "--silent", "--show-error", "--location", "--fail",
                             "--max-time", "120", url], capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Public download failed: host={urlparse(url).hostname}, exit={result.returncode}")
    return result.stdout


def get_json(url):
    return json.loads(get_bytes(url))


def main():
    directory = ROOT / "data/raw/external/hdb_20261019"
    directory.mkdir(parents=True, exist_ok=True)
    for name, dataset in DATASETS.items():
        target, receipt = directory / f"{name}.csv", directory / f"{name}_receipt.json"
        if target.exists() and receipt.exists():
            meta = json.loads(receipt.read_text())
            if hashlib.sha256(target.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError("Existing download checksum mismatch")
            print(f"VERIFIED {name} {meta['bytes']} bytes", flush=True)
            continue
        if target.exists() or receipt.exists():
            raise ValueError("Partial download: inspect before replacing")
        api = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset}"
        get_json(api + "/initiate-download")
        for attempt in range(5):
            result = get_json(api + "/poll-download")
            url = result.get("data", {}).get("url")
            if url:
                break
            time.sleep(3)
        else:
            raise RuntimeError("Download URL not ready; rerun later")
        if urlparse(url).scheme != "https":
            raise ValueError("Non-HTTPS download refused")
        content = get_bytes(url)
        if not content or content.lstrip().startswith(b"<"):
            raise ValueError("Empty or HTML download")
        temporary = target.with_suffix(".csv.part")
        temporary.write_bytes(content)
        temporary.replace(target)
        metadata = dict(dataset_id=dataset, source_page=f"https://data.gov.sg/datasets/{dataset}/view",
            retrieved_utc=datetime.now(timezone.utc).isoformat(), bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(), download_host=urlparse(url).hostname,
            licence="Singapore Open Data Licence; see source page",
            user_confirmed_external_data_allowed="2026-10-19",
            vintage="Latest snapshot, NOT a historical release archive",
            use=("Only event months <= training/query origin minus 2 months. Historical revisions remain unverified."
                 if name == "resale" else "Audit only; not training input without historic snapshot verification."))
        receipt.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print(f"DOWNLOADED {name} {len(content)} bytes", flush=True)


if __name__ == "__main__":
    main()
