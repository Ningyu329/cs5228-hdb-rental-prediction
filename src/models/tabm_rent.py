"""Official TabM backbone with numeric PLE and compact address embeddings.

Input customization follows the official custom-input recipe. Train the mean
of individual branch losses, never the loss of the averaged branch output.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np

from src.models.keras_rent import fit_schema, transform
from src.features.build_temporal_features import fit_trend, trend_values
from src.models.train_baseline import TARGET


def torch_api(device):
    import torch
    torch.set_num_threads(3)
    if device not in {"cpu", "mps"}:
        raise ValueError("Only local CPU or Apple MPS is supported in this experiment")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple GPU unavailable; use the same device as the saved experiment or a new output directory")
    return torch


def fit_tabm_schema(fit, kind, n_bins):
    schema = fit_schema(fit, kind)
    x = transform(schema, fit)["numeric"]
    # Keep constants and binary missing indicators as scalar inputs. Quantile
    # edges use training features only, with no target-dependent discretization.
    indices = [i for i in range(len(schema["nums"])) if np.unique(x[:, i]).size > 1]
    bins = [np.unique(np.quantile(x[:, i], np.linspace(0, 1, n_bins + 1))).astype("float32") for i in indices]
    kept = [(i, edges) for i, edges in zip(indices, bins) if len(edges) >= 2]
    schema.update(ple_indices=[i for i, _ in kept], bins=[e.tolist() for _, e in kept],
                  raw_indices=[i for i in range(x.shape[1]) if i not in {j for j, _ in kept}], n_bins=n_bins)
    return schema


def tensors(schema, frame, device):
    torch = torch_api(device)
    transformed = transform(schema, frame)
    cats = np.column_stack([transformed[f"category_{i}"].ravel() for i in range(len(schema["cats"]))])
    return (torch.as_tensor(transformed["numeric"], device=device),
            torch.as_tensor(cats, dtype=torch.long, device=device))


def make_network(schema, spec):
    import torch
    from torch import nn
    import tabm
    import rtdl_num_embeddings as ne

    class EmbeddedTabM(nn.Module):
        def __init__(self):
            super().__init__()
            self.ple = ne.PiecewiseLinearEmbeddings(
                [torch.tensor(b) for b in schema["bins"]], spec["d_embedding"], activation=False, version="B") if schema["bins"] else None
            dimensions = [min(16, max(2, int(np.ceil(np.sqrt(len(schema["levels"][c]) + 1)))) ) for c in schema["cats"]]
            self.categorical = nn.ModuleList([nn.Embedding(len(schema["levels"][c]) + 1, d)
                                              for c, d in zip(schema["cats"], dimensions)])
            chunks = [spec["d_embedding"]] * len(schema["ple_indices"]) + [1] * len(schema["raw_indices"]) + dimensions
            self.view = tabm.EnsembleView(k=spec["k"])
            self.backbone = tabm.make_tabm_backbone(d_in=sum(chunks), n_blocks=spec["n_blocks"],
                d_block=spec["d_block"], dropout=spec["dropout"], k=spec["k"], arch_type="tabm",
                start_scaling_init="normal", start_scaling_init_chunks=chunks)
            self.head = tabm.LinearEnsemble(spec["d_block"], 1, k=spec["k"])

        def forward(self, numeric, categorical):
            parts = []
            if self.ple is not None:
                parts.append(self.ple(numeric[:, schema["ple_indices"]]).flatten(1))
            if schema["raw_indices"]:
                parts.append(numeric[:, schema["raw_indices"]])
            parts.extend(emb(categorical[:, i]) for i, emb in enumerate(self.categorical))
            return self.head(self.backbone(self.view(torch.cat(parts, dim=-1)))).squeeze(-1)
    return EmbeddedTabM()


def independent_mse(branches, labels):
    if branches.ndim != 2 or labels.ndim != 1 or len(branches) != len(labels):
        raise ValueError("Expected (batch, branches) predictions and (batch,) labels")
    return ((branches - labels[:, None]) ** 2).mean()


@dataclass
class TabMRent:
    model: object
    schema: dict
    spec: dict
    trend: dict | None
    center: float
    scale: float
    device: str

    def response(self, frame):
        labels = frame[TARGET].to_numpy(dtype=float)
        if self.trend is not None:
            labels = labels - trend_values(self.trend, frame)
        return ((labels - self.center) / self.scale).astype("float32")

    def predict_scaled(self, x):
        torch = torch_api(self.device)
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x[0]), self.spec["evaluation_batch_size"]):
                end = start + self.spec["evaluation_batch_size"]
                predictions.append(self.model(x[0][start:end], x[1][start:end]).mean(dim=1).cpu().numpy())
        return np.concatenate(predictions).astype(float) if predictions else np.empty(0)

    def predict(self, frame):
        prediction = self.predict_scaled(tensors(self.schema, frame, self.device)) * self.scale + self.center
        if self.trend is not None:
            prediction += trend_values(self.trend, frame)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite TabM prediction")
        return np.maximum(prediction, 0.)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        torch = torch_api(self.device)
        torch.save({k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}, directory / "model.pt")
        (directory / "schema.json").write_text(json.dumps(dict(schema=self.schema, spec=self.spec,
            trend=self.trend, center=self.center, scale=self.scale, device=self.device),
            ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def load_bundle(directory, device=None):
    payload = json.loads((directory / "schema.json").read_text())
    if device is not None:
        payload["device"] = device
    torch = torch_api(payload["device"])
    if payload["trend"] is not None:
        payload["trend"]["monthly_levels"] = {int(k): v for k, v in payload["trend"]["monthly_levels"].items()}
    model = make_network(payload["schema"], payload["spec"])
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    return TabMRent(model.to(payload["device"]), **payload)


def train(fit, kind, seed, spec, valid=None, epochs=None, label="", device="cpu"):
    if valid is not None and (epochs is not None or fit.RENT_APPROVAL_DATE.max() >= valid.RENT_APPROVAL_DATE.min()):
        raise ValueError("Inner early-stopping window must be strictly later than the fitting rows")
    torch = torch_api(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    schema = fit_tabm_schema(fit, kind, spec["n_bins"])
    trend = fit_trend(fit, "damped24") if kind == "trend" else None
    labels = fit[TARGET].to_numpy(dtype=float)
    if trend is not None:
        labels = labels - trend_values(trend, fit)
    bundle = TabMRent(make_network(schema, spec).to(device), schema, spec, trend,
                      float(labels.mean()), max(float(labels.std()), 1.), device)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    x = tensors(schema, fit, device)
    y = torch.as_tensor(bundle.response(fit), device=device)
    vx = tensors(schema, valid, device) if valid is not None else None
    vy = bundle.response(valid).astype(float) if valid is not None else None
    generator = torch.Generator().manual_seed(seed)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(spec["max_epochs"] if epochs is None else epochs):
        bundle.model.train()
        permutation = torch.randperm(len(fit), generator=generator).to(device)
        epoch_loss = torch.zeros((), device=device)
        for indices in permutation.split(spec["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = independent_mse(bundle.model(x[0][indices], x[1][indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 1.)
            optimizer.step()
            epoch_loss += loss.detach() * len(indices)
        row = dict(epoch=epoch + 1, training_scaled_mse=float(epoch_loss.cpu()) / len(fit))
        if not np.isfinite(row["training_scaled_mse"]):
            raise ValueError("TabM training diverged")
        if valid is not None:
            row["validation_scaled_mse"] = float(np.mean((bundle.predict_scaled(vx) - vy) ** 2))
            if row["validation_scaled_mse"] < best:
                best, best_epoch = row["validation_scaled_mse"], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in bundle.model.state_dict().items()}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"EPOCH {label} {epoch + 1}: {row}; {time.perf_counter()-started:.1f}s", flush=True)
        if valid is not None and epoch + 1 - best_epoch >= spec["patience"]:
            break
    if valid is not None:
        if best_state is None:
            raise ValueError("No finite internal validation score")
        bundle.model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
    return bundle, dict(best_epoch=best_epoch, epochs_run=len(history), history=history,
        seconds=time.perf_counter() - started, parameter_count=sum(p.numel() for p in bundle.model.parameters()))
