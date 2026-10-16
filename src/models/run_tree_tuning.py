"""Bounded, resumable CatBoost/LightGBM tuning with temporal selection.

Housing features are fixed. Screen on Apr-Jun 2024; take one challenger per
family to Jul-Sep and Oct-Dec backtests. Lock models/weights before evaluating
Jan-Mar 2025. Never read test data or overwrite the previous stage's outputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.models.run_stage5 import FittedModel, feature_schema, load_training, metrics, grouped_errors
from src.models.run_location_ablation import validation_splits
from src.models.train_baseline import TARGET, split_metadata

EARLIER = ["temporal_backtest_2", "temporal_backtest_1"]
BASE_IDS = {"catboost": "catboost_original", "lightgbm": "lightgbm_original"}


def search_grid() -> dict[str, dict]:
    """Declare the complete search before looking at any new validation scores."""
    return {
        "catboost_original": dict(family="catboost", cap=400, patience=40,
            params=dict(depth=6, learning_rate=0.06, l2_leaf_reg=5.0)),
        "catboost_d6_long": dict(family="catboost", cap=1200, patience=80,
            params=dict(depth=6, learning_rate=0.04, l2_leaf_reg=5.0)),
        "catboost_d8_regularized": dict(family="catboost", cap=800, patience=80,
            params=dict(depth=8, learning_rate=0.05, l2_leaf_reg=10.0)),
        "lightgbm_original": dict(family="lightgbm", cap=400, patience=40,
            params=dict(num_leaves=31, learning_rate=0.04, min_child_samples=40, reg_lambda=5.0, colsample_bytree=0.9)),
        "lightgbm_31_slow": dict(family="lightgbm", cap=1200, patience=80,
            params=dict(num_leaves=31, learning_rate=0.02, min_child_samples=40, reg_lambda=5.0, colsample_bytree=0.9)),
        "lightgbm_63_regularized": dict(family="lightgbm", cap=1000, patience=80,
            params=dict(num_leaves=63, learning_rate=0.04, min_child_samples=80, reg_lambda=10.0, colsample_bytree=0.9)),
        "lightgbm_63_flexible": dict(family="lightgbm", cap=1000, patience=80,
            params=dict(num_leaves=63, learning_rate=0.04, min_child_samples=20, reg_lambda=5.0, colsample_bytree=0.9)),
        "lightgbm_15_regularized": dict(family="lightgbm", cap=1000, patience=80,
            params=dict(num_leaves=15, learning_rate=0.04, min_child_samples=80, reg_lambda=10.0, colsample_bytree=0.9)),
    }


def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_inner(fit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = fit.RENT_APPROVAL_DATE.dt.to_period("M")
    unique = sorted(months.unique())
    if len(unique) < 4:
        raise ValueError("At least four months are required")
    inner = months.isin(unique[-3:])
    return fit.loc[~inner].copy(), fit.loc[inner].copy()


def train_tree(spec: dict, fit: pd.DataFrame, threads: int,
               iterations: int | None = None, inner: pd.DataFrame | None = None) -> FittedModel:
    cats, nums = feature_schema("housing")
    family = spec["family"]
    levels = {column: fit[column].fillna("missing").astype(str).unique().tolist() for column in cats}
    bundle = FittedModel(family, None, cats, nums, levels)
    count = spec["cap"] if iterations is None else iterations
    if family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=count, **spec["params"], loss_function="RMSE",
            random_seed=42, thread_count=threads, allow_writing_files=False, verbose=False)
        kwargs = dict(cat_features=cats)
        if inner is not None:
            kwargs.update(eval_set=(bundle.matrix(inner), inner[TARGET]),
                early_stopping_rounds=spec["patience"], use_best_model=True)
    elif family == "lightgbm":
        from lightgbm import LGBMRegressor, early_stopping
        model = LGBMRegressor(n_estimators=count, **spec["params"], max_depth=-1,
            objective="regression", random_state=42, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True)
        kwargs = dict(categorical_feature=cats)
        if inner is not None:
            kwargs.update(eval_X=bundle.matrix(inner), eval_y=inner[TARGET], eval_metric="rmse",
                callbacks=[early_stopping(spec["patience"], verbose=False)])
    else:
        raise ValueError(f"Unknown family {family}")
    model.fit(bundle.matrix(fit), fit[TARGET], **kwargs)
    bundle.model = model
    return bundle


def chosen_rounds(bundle: FittedModel) -> int:
    return max(1, int(bundle.model.tree_count_ if bundle.family == "catboost" else bundle.model.best_iteration_))


def screen_one(candidate: str, spec: dict, fit: pd.DataFrame, inner: pd.DataFrame,
               cache: Path, threads: int) -> dict:
    directory = cache / "screen" / candidate
    done = directory / "result.json"
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    start = time.perf_counter()
    bundle = train_tree(spec, fit, threads, inner=inner)
    row = dict(candidate=candidate, family=spec["family"], **metrics(inner[TARGET], bundle.predict(inner)),
        iterations=chosen_rounds(bundle), cap=spec["cap"], seconds=time.perf_counter() - start,
        fit_end=fit.RENT_APPROVAL_DATE.max().strftime("%Y-%m"),
        validation_start=inner.RENT_APPROVAL_DATE.min().strftime("%Y-%m"),
        validation_end=inner.RENT_APPROVAL_DATE.max().strftime("%Y-%m"))
    row["hit_cap"] = row["iterations"] == spec["cap"]
    directory.mkdir(parents=True, exist_ok=True)
    dump_json(done, row)
    return row


def shortlist(screen: pd.DataFrame, grid: dict) -> dict[str, str]:
    # Always test the best *new* setting against the old family baseline outside
    # the screening window; the old setting remains eligible at final selection.
    selected = {}
    for family in BASE_IDS:
        eligible = screen.loc[screen.family.eq(family) & ~screen.candidate.eq(BASE_IDS[family])]
        if eligible.empty:
            raise ValueError(f"No challenger for {family}")
        name = str(eligible.sort_values(["rmse", "candidate"]).iloc[0].candidate)
        if grid[name]["family"] != family:
            raise ValueError("Family mismatch")
        selected[family] = name
    return selected


def nested_one(candidate: str, spec: dict, name: str, fit: pd.DataFrame, query: pd.DataFrame,
               cache: Path, threads: int, reused_screen: dict | None = None) -> tuple[np.ndarray, dict]:
    directory = cache / name / candidate
    done = directory / "result.json"
    if done.is_file():
        details = json.loads(done.read_text(encoding="utf-8"))
        prediction = np.load(directory / "prediction.npy")
        if len(prediction) != len(query) or not np.isfinite(prediction).all():
            raise ValueError("Invalid cached predictions")
        return prediction, details
    start = time.perf_counter()
    inner_fit, inner_valid = split_inner(fit)
    if reused_screen is not None:
        if reused_screen["validation_end"] != inner_valid.RENT_APPROVAL_DATE.max().strftime("%Y-%m"):
            raise ValueError("Screen and nested early-stopping windows do not match")
        rounds = int(reused_screen["iterations"])
    else:
        tuned = train_tree(spec, inner_fit, threads, inner=inner_valid)
        rounds = chosen_rounds(tuned)
    # query/TARGET never enters training, early stopping or categorical levels.
    bundle = train_tree(spec, fit, threads, iterations=rounds)
    prediction = bundle.predict(query)
    details = dict(iterations=rounds, cap=spec["cap"], hit_cap=rounds == spec["cap"],
        inner_fit_end=inner_fit.RENT_APPROVAL_DATE.max().strftime("%Y-%m"),
        inner_validation_start=inner_valid.RENT_APPROVAL_DATE.min().strftime("%Y-%m"),
        inner_validation_end=inner_valid.RENT_APPROVAL_DATE.max().strftime("%Y-%m"),
        seconds=time.perf_counter() - start)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "prediction.npy", prediction)
    if name == "temporal":
        if spec["family"] == "catboost":
            bundle.model.save_model(str(directory / "model.cbm"))
        else:
            bundle.model.booster_.save_model(str(directory / "model.txt"))
        dump_json(directory / "schema.json", dict(family=bundle.family, cats=bundle.cats,
            nums=bundle.nums, levels=bundle.levels))
    dump_json(done, details)
    return prediction, details


def blend_menu(challengers: dict[str, str]) -> dict[str, dict[str, float]]:
    cats = [BASE_IDS["catboost"], challengers["catboost"]]
    lights = [BASE_IDS["lightgbm"], challengers["lightgbm"]]
    menu = {name: {name: 1.0} for name in cats + lights}
    for cat in cats:
        for light in lights:
            for weight in [0.25, 0.5, 0.75]:
                name = f"blend__{cat}__{light}__cat{int(weight * 100)}"
                menu[name] = {cat: weight, light: 1 - weight}
    return menu


def combine(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not weights or any(value < 0 for value in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Blend weights must be nonnegative and sum to one")
    return sum(predictions[name] * weight for name, weight in weights.items())


def choose_locked(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    earlier = results.loc[results.split.isin(EARLIER)].copy()
    if earlier.empty or earlier.duplicated(["candidate", "split"]).any():
        raise ValueError("Need unique paired earlier backtests")
    if not earlier.groupby("candidate").split.agg(set).map(lambda names: names == set(EARLIER)).all():
        raise ValueError("Every candidate needs both earlier windows")
    earlier["sse"] = earlier.rmse ** 2 * earlier.validation_rows
    table = earlier.groupby("candidate").agg(sse=("sse", "sum"), rows=("validation_rows", "sum")).reset_index()
    table["pooled_rmse"] = np.sqrt(table.sse / table.rows)
    table = table.sort_values(["pooled_rmse", "candidate"]).reset_index(drop=True)
    return str(table.iloc[0].candidate), table


def load_reference(directory: Path, split: str, query: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = pd.read_csv(directory / f"predictions_{split}.csv")
    if not np.array_equal(frame.source_row.to_numpy(), query.index.to_numpy()):
        raise ValueError("Previous stage row order differs")
    np.testing.assert_allclose(frame.actual, query[TARGET], rtol=0, atol=1e-8)
    result = {BASE_IDS[family]: frame[f"{family}_housing"].to_numpy() for family in BASE_IDS}
    if not all(np.isfinite(pred).all() for pred in result.values()):
        raise ValueError("Non-finite reference predictions")
    return result


def report_text(results: pd.DataFrame, screen: pd.DataFrame, lock: dict, comparison: pd.DataFrame) -> str:
    lines = ["# CatBoost / LightGBM temporal tuning", "", "## Design", "",
        "- Fixed housing feature set: no new target statistics, MRT, macro data, test data or label removal.",
        "- Screen all eight configurations on Apr-Jun 2024, training through Mar 2024. Carry one new configuration per family forward.",
        "- Compare challengers, original components and 25/50/75% CatBoost blends on Jul-Sep and Oct-Dec 2024. Choose by pooled squared error.",
        "- The original 50/50 blend remains eligible. Both earlier windows are tuning data, not unbiased final performance estimates.",
        "- Each refit uses an internal three-month early-stopping window, never outer-query labels. The first screening fit is reused only for the identical inner window.",
        "- Lock before Jan-Mar 2025 evaluation. This primary period has been inspected in earlier project work and is not a pristine final test.",
        "- Search is bounded, not exhaustive; increased caps/depths do not guarantee improvement. No competition submission is generated.", "",
        "## Screening", "", "| Candidate | RMSE | Rounds | Cap |", "| --- | ---: | ---: | ---: |"]
    for row in screen.sort_values("rmse").itertuples():
        lines.append(f"| {row.candidate} | {row.rmse:.4f} | {row.iterations} | {row.cap} |")
    lines += ["", "## Locked selection", "", f"Selected: `{lock['selected_candidate']}`.", "",
        "```json", json.dumps(lock["weights"], indent=2), "```", "", "## Before / after", "",
        "| Model | Earlier pooled RMSE | Primary RMSE | Primary MAE | Primary bias |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.model} | {row.pooled_backtest_rmse:.4f} | {row.rmse:.4f} | {row.mae:.4f} | {row.bias:+.2f} |")
    old = comparison.loc[comparison.model.eq("Original 50/50 blend")].iloc[0]
    new = comparison.loc[comparison.model.eq("Locked selection")].iloc[0]
    lines += ["", f"Primary RMSE change (old minus selected): {old.rmse - new.rmse:+.4f} ({(old.rmse - new.rmse) / old.rmse:+.2%}).",
        "A negative change means the locked tuning choice did not improve the primary window; do not reselect based on that observation.",
        "", "## Limits and reproducibility", "",
        "- Static HDB snapshot availability limitations from stage 5 remain. Missing apartment-level details and long-horizon market uncertainty are unchanged.",
        "- A model selected at its iteration cap may still benefit from a later higher-cap experiment; this run makes no global-optimum claim.",
        "- Random splits are deliberately omitted: this experiment targets future-month prediction.",
        "- Saved primary models are diagnostic fits through Dec 2024, not final full-data models.",
        "- See experiment_manifest.json for parameters, input/code hashes, library versions, fixed search/selection rules, and reference artifact hashes.",
        "- See screening.csv, backtest_results.csv, selection_backtests.csv, selection_locked.json, results.csv, comparison.csv and group_errors.csv.",
        "", "[CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning) · [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)", ""]
    return "\n".join(lines)


def run(output: Path, threads: int = 4, workers: int = 3) -> dict:
    import catboost
    import lightgbm
    import sklearn
    if threads < 1 or workers < 1:
        raise ValueError("threads and workers must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/train_location_features.csv"
    reference = ROOT / "reports/experiments/stage5_v1"
    previous = json.loads((reference / "experiment_manifest.json").read_text(encoding="utf-8"))
    if digest(source) != next(value for key, value in previous["source_sha256"].items() if key.replace("\\", "/") == "data/processed/train_location_features.csv"):
        raise ValueError("Stage-5 reference training data has changed")
    grid = search_grid()
    paths = [Path(__file__), ROOT / "src/models/run_stage5.py", ROOT / "src/models/run_location_ablation.py",
        ROOT / "src/models/train_baseline.py", ROOT / "src/features/build_stage5_features.py"]
    for path in paths[1:]:
        key = str(path.relative_to(ROOT))
        if key in previous["implementation_sha256"] and digest(path) != previous["implementation_sha256"][key]:
            raise ValueError(f"Stage-5 reference implementation changed: {key}")
    reference_paths = [reference / f"predictions_{name}.csv" for name in EARLIER + ["temporal"]]
    manifest = dict(environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, catboost=catboost.__version__, lightgbm=lightgbm.__version__),
        train_sha256=digest(source), implementation_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
        reference_sha256={str(path.relative_to(ROOT)): digest(path) for path in reference_paths + [reference / "experiment_manifest.json"]},
        grid=grid, threads=threads, workers=workers, seed=42, schema=dict(zip(["categorical", "numeric"], feature_schema("housing"))),
        screen="2024-04..2024-06; fit <=2024-03; best new configuration per family",
        selection="lowest pooled SSE on Jul-Sep and Oct-Dec 2024; fixed 25/50/75% CatBoost weights; original pair and single models eligible",
        primary_used_for_selection=False, primary_previously_inspected=True,
        test_data_read=False, final_submission_generated=False)
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache = ROOT / "models/tree_tuning_cache" / signature
    manifest["cache_directory"] = str(cache.relative_to(ROOT))
    if any(output.iterdir()):
        old_manifest = output / "experiment_manifest.json"
        if not old_manifest.is_file() or json.loads(old_manifest.read_text(encoding="utf-8")).get("cache_directory") != manifest["cache_directory"]:
            raise ValueError("Output belongs to a different run; use a new output directory")
    dump_json(output / "experiment_manifest.json", manifest)
    train = load_training(source)
    splits = {name: (fit, query) for name, fit, query in validation_splits(train, 3, 2, False)}
    fit_screen, valid_screen = split_inner(splits[EARLIER[0]][0])
    rows = []
    print("SCREEN: 8 configurations on Apr-Jun 2024; housing features fixed", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Cheap LightGBM jobs first, then the CPU-heavy CatBoost fits.
        jobs = {pool.submit(screen_one, name, spec, fit_screen, valid_screen, cache, threads): name
            for name, spec in sorted(grid.items(), key=lambda pair: pair[1]["family"] == "catboost")}
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "screening.csv", index=False)
            print(f"SCREEN {row['candidate']}: {row['rmse']:.4f}; rounds={row['iterations']}/{row['cap']}; {row['seconds']:.1f}s", flush=True)
    screen = pd.DataFrame(rows)
    challengers = shortlist(screen, grid)
    dump_json(output / "shortlist_locked.json", dict(challengers=challengers, screen_only=True))
    print(f"SHORTLIST {challengers}", flush=True)
    menu = blend_menu(challengers)
    records, predictions_by_split = [], {}
    for name in EARLIER:
        fit, query = splits[name]
        predictions = load_reference(reference, name, query)
        metadata = split_metadata(name, fit, query)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(nested_one, candidate, grid[candidate], name, fit, query, cache, threads,
                screen.loc[screen.candidate.eq(candidate)].iloc[0].to_dict() if name == EARLIER[0] else None): candidate
                for candidate in challengers.values()}
            for future in as_completed(jobs):
                candidate = jobs[future]
                prediction, details = future.result()
                predictions[candidate] = prediction
                print(f"{name} {candidate}: {metrics(query[TARGET], prediction)['rmse']:.4f}; rounds={details['iterations']}; {details['seconds']:.1f}s", flush=True)
                dump_json(output / f"{name}_{candidate}_fit.json", details)
        for candidate, weights in menu.items():
            pred = combine(predictions, weights)
            predictions[candidate] = pred
            records.append(dict(**metadata, candidate=candidate, **metrics(query[TARGET], pred)))
        predictions_by_split[name] = predictions
        pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy(), **predictions)).to_csv(output / f"predictions_{name}.csv", index=False)
        pd.DataFrame(records).to_csv(output / "backtest_results.csv", index=False)
    selected, ranking = choose_locked(pd.DataFrame(records))
    ranking.to_csv(output / "selection_backtests.csv", index=False)
    lock = dict(selected_candidate=selected, weights=menu[selected], challengers=challengers,
        selection_splits=EARLIER, primary_used=False, primary_previously_inspected=True)
    dump_json(output / "selection_locked.json", lock)
    print(f"LOCKED before primary: {selected}; {menu[selected]}", flush=True)
    # Only after writing the lock do we access or score primary predictions.
    fit, query = splits["temporal"]
    predictions = load_reference(reference, "temporal", query)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(nested_one, candidate, grid[candidate], "temporal", fit, query, cache, threads): candidate
            for candidate in challengers.values()}
        for future in as_completed(jobs):
            candidate = jobs[future]
            pred, details = future.result()
            predictions[candidate] = pred
            dump_json(output / f"temporal_{candidate}_fit.json", details)
            print(f"PRIMARY {candidate}: {metrics(query[TARGET], pred)['rmse']:.4f}; rounds={details['iterations']}; {details['seconds']:.1f}s", flush=True)
    # Report predeclared individual challengers and locked mixture only, not all
    # primary weight combinations (which would invite post-hoc reselection).
    original = {BASE_IDS["catboost"]: 0.5, BASE_IDS["lightgbm"]: 0.5}
    predictions["original_blend"] = combine(predictions, original)
    predictions["locked_selection"] = combine(predictions, lock["weights"])
    for candidate, pred in predictions.items():
        records.append(dict(**split_metadata("temporal", fit, query), candidate=candidate, **metrics(query[TARGET], pred)))
    pd.DataFrame(dict(source_row=query.index, actual=query[TARGET].to_numpy(), **predictions)).to_csv(output / "predictions_temporal.csv", index=False)
    results = pd.DataFrame(records)
    results.to_csv(output / "results.csv", index=False)
    comparison_rows = []
    comparisons = [("Original CatBoost", {BASE_IDS["catboost"]: 1}),
        ("Tuned CatBoost", {challengers["catboost"]: 1}),
        ("Original LightGBM", {BASE_IDS["lightgbm"]: 1}),
        ("Tuned LightGBM", {challengers["lightgbm"]: 1}),
        ("Original 50/50 blend", original), ("Locked selection", lock["weights"])]
    for label, weights in comparisons:
        sse, count = 0.0, 0
        for name in EARLIER:
            actual = splits[name][1][TARGET].to_numpy()
            sse += float(np.sum((combine(predictions_by_split[name], weights) - actual) ** 2))
            count += len(actual)
        comparison_rows.append(dict(model=label, pooled_backtest_rmse=np.sqrt(sse / count),
            **metrics(query[TARGET], combine(predictions, weights))))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    grouped_errors(query, {name: predictions[name] for name in ["original_blend", "locked_selection", *challengers.values()]}).to_csv(output / "group_errors.csv", index=False)
    (output / "tuning_report.md").write_text(report_text(results, screen, lock, comparison), encoding="utf-8")
    manifest["selection"] = lock
    manifest["diagnostic_training_end"] = "2024-12"
    dump_json(output / "experiment_manifest.json", manifest)
    print(comparison.to_string(index=False), flush=True)
    print(f"DONE {output}", flush=True)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/tree_tuning_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.output_dir, args.threads, args.workers)
