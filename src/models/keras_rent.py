"""Small Keras/JAX tabular regressor; fit-only preprocessing and target scaling."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.features.build_temporal_features import fit_trend, trend_values
from src.models.run_stage5 import feature_schema
from src.models.train_baseline import TARGET

ROOT = Path(__file__).resolve().parents[2]
TIME_NUMS = {"RENT_YEAR", "RENT_MONTH", "RENT_MONTH_INDEX", "FLAT_AGE", "BUILDING_AGE",
             "RENT_MONTH_SIN", "RENT_MONTH_COS"}


def keras_api():
    # No user-level Keras, font or compilation-cache files are written.
    runtime = ROOT / "models/keras_runtime"
    os.environ.setdefault("KERAS_BACKEND", "jax")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("KERAS_HOME", str(runtime / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime / "cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "3")
    import keras
    if keras.backend.backend() != "jax":
        raise ValueError("This reproducible experiment requires the JAX backend")
    return keras


def recipe_schema(kind):
    if kind not in {"direct", "trend"}:
        raise ValueError("Unknown Keras recipe")
    cats, nums = feature_schema("housing")
    if kind == "trend":
        cats = [c for c in cats if c != "FLAT_AGE_BAND"]
        nums = [c for c in nums if c not in TIME_NUMS]
    if {TARGET, "Id", "source_row"}.intersection(cats + nums):
        raise ValueError("Target/ID must never be a predictor")
    return cats, nums


def numeric_values(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def fit_schema(fit, kind):
    if fit.empty:
        raise ValueError("Empty fitting data")
    cats, nums = recipe_schema(kind)
    levels = {}
    for col in cats:
        counts = fit[col].fillna("missing").astype(str).value_counts()
        # Singletons share the same trained bucket as unseen categories (index 0).
        levels[col] = sorted(counts.index[counts.ge(2)].tolist())
    x = numeric_values(fit, nums)
    medians = [float(np.median(x[np.isfinite(x[:, i]), i])) if np.isfinite(x[:, i]).any() else 0.
               for i in range(len(nums))]
    filled = np.where(np.isfinite(x), x, medians)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.)
    return dict(cats=cats, nums=nums, levels=levels, medians=medians,
                means=means.tolist(), scales=scales.tolist(), kind=kind,
                numeric_missing_indicators=True, rare_min_count=2)


def transform(schema, frame):
    # Only schema-named predictors are read, never labels, IDs or row order keys.
    x = numeric_values(frame, schema["nums"])
    missing = ~np.isfinite(x)
    filled = np.where(missing, schema["medians"], x)
    standardized = (filled - schema["means"]) / schema["scales"]
    out = {"numeric": np.column_stack([standardized, missing]).astype("float32")}
    for i, col in enumerate(schema["cats"]):
        lookup = {value: j + 1 for j, value in enumerate(schema["levels"][col])}
        out[f"category_{i}"] = frame[col].fillna("missing").astype(str).map(lookup).fillna(0).to_numpy(
            dtype="int32").reshape(-1, 1)
    if not np.isfinite(out["numeric"]).all():
        raise ValueError("Non-finite numeric network input")
    return out


def build_network(schema, seed, spec):
    keras = keras_api()
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    numerical = keras.Input((2 * len(schema["nums"]),), name="numeric", dtype="float32")
    inputs, encoded = {"numeric": numerical}, [numerical]
    for i, col in enumerate(schema["cats"]):
        key = f"category_{i}"
        inputs[key] = keras.Input((1,), name=key, dtype="int32")
        cardinality = len(schema["levels"][col]) + 1
        dimension = min(16, max(2, int(np.ceil(np.sqrt(cardinality)))))
        vector = keras.layers.Embedding(cardinality, dimension,
            embeddings_regularizer=keras.regularizers.L2(spec["embedding_l2"]),
            name=f"embedding_{i}")(inputs[key])
        encoded.append(keras.layers.Flatten()(vector))
    x = keras.layers.Concatenate()(encoded)
    for width in spec["widths"]:
        x = keras.layers.Dense(width, activation="relu",
            kernel_regularizer=keras.regularizers.L2(spec["dense_l2"]))(x)
        x = keras.layers.Dropout(spec["dropout"])(x)
    output = keras.layers.Dense(1, kernel_initializer="zeros", name="scaled_rent")(x)
    model = keras.Model(inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=spec["learning_rate"], clipnorm=5.),
                  loss="mean_squared_error", metrics=[keras.metrics.MeanSquaredError(name="scaled_mse")],
                  jit_compile=True)
    return model


@dataclass
class KerasRent:
    model: object
    schema: dict
    trend: dict | None
    center: float
    scale: float

    def response(self, frame):
        y = frame[TARGET].to_numpy(dtype=float)
        if self.trend is not None:
            y = y - trend_values(self.trend, frame)
        return ((y - self.center) / self.scale).astype("float32").reshape(-1, 1)

    def predict(self, frame):
        if frame.empty:
            return np.empty(0, dtype=float)
        # Fixed batch size avoids compiling a large full-frame prediction graph.
        raw = np.asarray(self.model.predict(transform(self.schema, frame), batch_size=2048, verbose=0),
                         dtype=float).ravel()
        pred = raw * self.scale + self.center
        if self.trend is not None:
            pred += trend_values(self.trend, frame)
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite Keras prediction")
        return np.maximum(pred, 0.)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        (directory / "schema.json").write_text(json.dumps(dict(schema=self.schema, trend=self.trend,
            center=self.center, scale=self.scale), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def load_bundle(directory):
    payload = json.loads((directory / "schema.json").read_text())
    if payload["trend"] is not None:
        payload["trend"]["monthly_levels"] = {int(k): v for k, v in payload["trend"]["monthly_levels"].items()}
    return KerasRent(keras_api().models.load_model(directory / "model.keras", compile=False), **payload)


def train(fit, kind, seed, spec, valid=None, epochs=None, label=""):
    if valid is not None and (epochs is not None or fit.RENT_APPROVAL_DATE.max() >= valid.RENT_APPROVAL_DATE.min()):
        raise ValueError("Early stopping needs a strictly later inner window and no fixed epoch count")
    keras = keras_api()
    schema = fit_schema(fit, kind)
    trend = fit_trend(fit, "damped24") if kind == "trend" else None
    labels = fit[TARGET].to_numpy(dtype=float)
    if trend is not None:
        labels = labels - trend_values(trend, fit)
    center, scale = float(labels.mean()), max(float(labels.std()), 1.)
    bundle = KerasRent(build_network(schema, seed, spec), schema, trend, center, scale)
    callbacks = [keras.callbacks.TerminateOnNaN()]
    if valid is not None:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_scaled_mse", patience=spec["patience"],
            min_delta=0., restore_best_weights=True))

    class Progress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                numbers = ", ".join(f"{k}={v:.5f}" for k, v in (logs or {}).items())
                print(f"EPOCH {label} {epoch + 1}: {numbers}", flush=True)
    callbacks.append(Progress())
    started = time.perf_counter()
    history = bundle.model.fit(transform(schema, fit), bundle.response(fit),
        validation_data=None if valid is None else (transform(schema, valid), bundle.response(valid)),
        epochs=spec["max_epochs"] if epochs is None else epochs, batch_size=spec["batch_size"],
        shuffle=True, callbacks=callbacks, verbose=0).history
    if not all(np.isfinite(v).all() for v in history.values()):
        raise ValueError("Keras training diverged")
    best = int(np.argmin(history["val_scaled_mse"])) + 1 if valid is not None else len(history["loss"])
    return bundle, dict(best_epoch=best, epochs_run=len(history["loss"]), history=history,
                        seconds=time.perf_counter() - started, parameter_count=bundle.model.count_params())


def release(bundle):
    del bundle
    keras_api().backend.clear_session()
    gc.collect()
