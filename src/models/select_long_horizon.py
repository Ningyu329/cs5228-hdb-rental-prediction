"""Separate long-horizon selection: two early origins, then a later-origin check."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from src.models.run_rmse_improvement import dump, sha

EARLY = ["origin_202212_16m", "origin_202306_16m"]
CHECK = "origin_202311_16m"
CANDIDATES = ["housing_lgb", "resale_lgb_only", "resale_lgb_blend25", "hierarchy_only", "hierarchy_blend25",
              "hierarchy_residual_lgb_only", "hierarchy_residual_lgb_blend25"]


def select(rows):
    reference = {r["split"]: r["rmse"] for r in rows if r["split"] in EARLY and r["candidate"] == "housing_lgb"}
    ranked = []
    for name in CANDIDATES:
        group = [r for r in rows if r["split"] in EARLY and r["candidate"] == name]
        if len(group) != 2 or {r["split"] for r in group} != set(EARLY):
            raise ValueError("Two complete early long-horizon origins required")
        rmse = float(np.sqrt(sum(r["rmse"] ** 2 * r["rows"] for r in group) / sum(r["rows"] for r in group)))
        ranked.append(dict(candidate=name, rmse=rmse,
                           worst_origin_delta=max(r["rmse"] - reference[r["split"]] for r in group)))
    baseline = next(r["rmse"] for r in ranked if r["candidate"] == "housing_lgb")
    ranked.sort(key=lambda r: (r["rmse"], r["candidate"]))
    for r in ranked:
        r["gain"] = baseline - r["rmse"]
        r["eligible"] = r["gain"] >= 5 and r["worst_origin_delta"] <= 10
    eligible = [r for r in ranked if r["eligible"]]
    return ranked, eligible[0]["candidate"] if eligible else "housing_lgb"


def main(output, plan_only=False):
    plan = dict(implementation_sha256=sha(Path(__file__)), selection_origins=EARLY, confirmation_origin=CHECK,
        candidates=CANDIDATES, min_pooled_gain=5, max_origin_degradation=10, anchor="standalone housing LightGBM, NOT production blend",
        production_replacement=False, interpretation="Repeated forecast tasks; overlapping contracts/origins are not independent samples",
        primary_and_latest_origin_used_for_selection=False)
    path = output / "long_selection_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Predeclared long plan changed")
    if not path.exists():
        if (output / "completion.json").exists():
            raise ValueError("Cannot first declare the plan after completion")
        dump(path, plan)
    if plan_only:
        print("Long-horizon selection declared before long results")
        return
    rows = json.loads((output / "results.json").read_text())
    ranked, selected = select(rows)
    confirmation = [r for r in rows if r["split"] == CHECK and r["candidate"] in ["housing_lgb", selected]]
    if len(confirmation) != (1 if selected == "housing_lgb" else 2):
        raise ValueError("Latest-origin confirmation missing/duplicated")
    dump(output / "long_selection.json", dict(selected=selected, ranking=ranked, confirmation=confirmation,
        primary_and_latest_origin_used_for_selection=False, production_replaced=False, plan_sha256=sha(path)))
    print(json.dumps(dict(selected=selected, confirmation=confirmation), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/data_rethink_v3")
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.plan_only)
