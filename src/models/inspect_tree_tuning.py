"""Independently verify tuning artifacts and portable diagnostic model reloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from src.models.run_stage5 import FittedModel, load_training, metrics
from src.models.run_tree_tuning import (
    BASE_IDS, EARLIER, blend_menu, choose_locked, combine, digest,
    load_reference, shortlist, split_inner,
)
from src.models.run_location_ablation import validation_splits


def inspect(output: Path) -> dict:
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((output / "selection_locked.json").read_text(encoding="utf-8"))
    assert manifest["selection"] == lock
    assert lock["primary_used"] is False
    assert manifest["test_data_read"] is False
    for field in ["implementation_sha256", "reference_sha256"]:
        for name, expected in manifest[field].items():
            assert digest(ROOT / name) == expected, f"Changed artifact: {name}"
    source = ROOT / "data/processed/train_location_features.csv"
    assert digest(source) == manifest["train_sha256"]
    train = load_training(source)
    splits = {name: (fit, query) for name, fit, query in validation_splits(train, 3, 2, False)}
    screen = pd.read_csv(output / "screening.csv")
    assert len(screen) == len(manifest["grid"]) == 8
    assert not screen.candidate.duplicated().any()
    assert shortlist(screen, manifest["grid"]) == lock["challengers"]
    assert screen.fit_end.eq("2024-03").all() and screen.validation_end.eq("2024-06").all()
    menu = blend_menu(lock["challengers"])
    assert lock["weights"] == menu[lock["selected_candidate"]]
    results = pd.read_csv(output / "results.csv")
    backtests = pd.read_csv(output / "backtest_results.csv")
    selected, ranking = choose_locked(backtests)
    assert selected == lock["selected_candidate"]
    exported_ranking = pd.read_csv(output / "selection_backtests.csv")
    assert ranking.candidate.tolist() == exported_ranking.candidate.tolist()
    np.testing.assert_allclose(ranking.pooled_rmse, exported_ranking.pooled_rmse, rtol=0, atol=1e-8)
    assert len(backtests) == 2 * len(menu)
    assert not results.duplicated(["split", "candidate"]).any()
    checked = 0
    prediction_tables = {}
    for name in EARLIER + ["temporal"]:
        fit, query = splits[name]
        frame = pd.read_csv(output / f"predictions_{name}.csv")
        prediction_tables[name] = frame
        np.testing.assert_array_equal(frame.source_row, query.index)
        np.testing.assert_allclose(frame.actual, query.MONTHLY_RENT, rtol=0, atol=1e-8)
        assert frame.drop(columns=["source_row"]).map(np.isfinite).all().all()
        originals = load_reference(ROOT / "reports/experiments/stage5_v1", name, query)
        for candidate, prediction in originals.items():
            np.testing.assert_allclose(frame[candidate], prediction, rtol=0, atol=1e-8)
        if name in EARLIER:
            for candidate, weights in menu.items():
                np.testing.assert_allclose(frame[candidate], combine(frame, weights), rtol=0, atol=1e-8)
        else:
            np.testing.assert_allclose(frame.locked_selection, combine(frame, lock["weights"]), rtol=0, atol=1e-8)
            np.testing.assert_allclose(frame.original_blend, (frame.catboost_original + frame.lightgbm_original) / 2, rtol=0, atol=1e-8)
        for row in results.loc[results.split.eq(name)].itertuples():
            assert row.validation_rows == len(query)
            measured = metrics(query.MONTHLY_RENT, frame[row.candidate].to_numpy())
            np.testing.assert_allclose([row.rmse, row.mae, row.bias], list(measured.values()), rtol=0, atol=1e-8)
            checked += len(query)
        inner_fit, inner = split_inner(fit)
        for candidate in lock["challengers"].values():
            details = json.loads((output / f"{name}_{candidate}_fit.json").read_text(encoding="utf-8"))
            assert details["inner_fit_end"] == inner_fit.RENT_APPROVAL_DATE.max().strftime("%Y-%m")
            assert details["inner_validation_start"] == inner.RENT_APPROVAL_DATE.min().strftime("%Y-%m")
            assert details["inner_validation_end"] == inner.RENT_APPROVAL_DATE.max().strftime("%Y-%m")
            assert details["inner_fit_end"] < details["inner_validation_start"] <= details["inner_validation_end"] < query.RENT_APPROVAL_DATE.min().strftime("%Y-%m")
            assert 1 <= details["iterations"] <= details["cap"]
    primary = splits["temporal"][1]
    comparison = pd.read_csv(output / "comparison.csv")
    original_weights = {BASE_IDS["catboost"]: 0.5, BASE_IDS["lightgbm"]: 0.5}
    comparison_weights = {
        "Original CatBoost": {BASE_IDS["catboost"]: 1.0},
        "Tuned CatBoost": {lock["challengers"]["catboost"]: 1.0},
        "Original LightGBM": {BASE_IDS["lightgbm"]: 1.0},
        "Tuned LightGBM": {lock["challengers"]["lightgbm"]: 1.0},
        "Original 50/50 blend": original_weights,
        "Locked selection": lock["weights"],
    }
    for row in comparison.itertuples():
        weights = comparison_weights[row.model]
        sse, count = 0.0, 0
        for name in EARLIER:
            table = prediction_tables[name]
            sse += float(np.sum((combine(table, weights) - table.actual) ** 2))
            count += len(table)
        expected = metrics(primary.MONTHLY_RENT, np.asarray(combine(prediction_tables["temporal"], weights)))
        np.testing.assert_allclose([row.pooled_backtest_rmse, row.rmse, row.mae, row.bias],
            [np.sqrt(sse / count), *expected.values()], rtol=0, atol=1e-8)
    reloads = {}
    for candidate in lock["challengers"].values():
        directory = ROOT / manifest["cache_directory"] / "temporal" / candidate
        schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
        if schema["family"] == "catboost":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(directory / "model.cbm"))
        else:
            from lightgbm import Booster
            model = Booster(model_file=str(directory / "model.txt"))
        bundle = FittedModel(model=model, **schema)
        details = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert (model.tree_count_ if schema["family"] == "catboost" else model.current_iteration()) == details["iterations"]
        np.testing.assert_allclose(bundle.predict(primary), prediction_tables["temporal"][candidate], rtol=0, atol=1e-7)
        reloads[candidate] = "matched all primary rows"
    groups = pd.read_csv(output / "group_errors.csv")
    for (candidate, field), group in groups.groupby(["candidate", "field"]):
        assert group.rows.sum() == len(primary)
        weighted_rmse = np.sqrt((group.rmse ** 2 * group.rows).sum() / group.rows.sum())
        expected = metrics(primary.MONTHLY_RENT, prediction_tables["temporal"][candidate].to_numpy())
        np.testing.assert_allclose(weighted_rmse, expected["rmse"], rtol=0, atol=1e-8)
    validation = dict(status="passed", result_rows=len(results), screening_configs=len(screen),
        checked_prediction_values=checked, portable_model_reload=reloads,
        input_and_code_hashes="matched", primary_selection="excluded by selection function and locked artifact",
        group_error_rows=len(groups), test_data_read=False)
    (output / "output_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/experiments/tree_tuning_v1")
    inspect(parser.parse_args().output_dir)
