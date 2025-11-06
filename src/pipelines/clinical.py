#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# repo-local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loss import cox_loss
from utils.split import (
    load_clinical_data_for_stratification,
    confirm_split_accuracy,
    confirm_split_feature_balance,
    build_composite_strat_labels,
)
from utils.preprocess import create_clinical_preprocessor
from utils.datasets import ClinicalDataset
from utils.models.mlp import PredictionModel_Clinical
from utils.metrics import compute_cindex, append_result
from utils.output import get_results_dir, write_metrics_csv, write_config_json


# --------------------------
# Config
# --------------------------
@dataclass
class CFG:
    project: str = "Clinical-Survival-Analysis-Stratified"
    seed: int = 42
    n_splits: int = 10
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 5e-5
    hidden_dim: int = 64
    dropout: float = 0.4
    data_dir: str = "data"
    results_scope: str = "clinical"
    resources_dir: str = "resources"
    survival_time_col: str = "Time_to_prog_or_FUend"
    survival_event_col: str = "progression"
    wandb_watch: bool = False
    log_freq: int = 5


def set_seed(seed: int) -> None:
    """Reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------
# Data loading helpers
# --------------------------
def load_clinical_dataframe(data_dir: str, patient_ids: List[str]) -> pd.DataFrame:
    """Aggregate per-patient *_CD.json files into a DataFrame indexed by pid."""
    rows: List[Dict] = []
    for pid in patient_ids:
        cd_path = os.path.join(data_dir, pid, f"{pid}_CD.json")
        with open(cd_path, "r") as f:
            row = json.load(f)
        row["pid"] = pid
        rows.append(row)
    df = pd.DataFrame(rows).set_index("pid")
    df.replace(-1, np.nan, inplace=True)
    return df


def make_cv_splits(cfg: CFG, all_pids: List[str], df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build composite strat labels and return all (train, val) index splits."""
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    strat_labels = build_composite_strat_labels(
        df=df,
        patient_ids=all_pids,
        n_splits=cfg.n_splits,
        progression_col=cfg.survival_event_col,
        stage_col="stage",
        substage_col="substage",
        min_count=cfg.n_splits,
    )
    return list(skf.split(all_pids, strat_labels))


# --------------------------
# Training/Eval per fold
# --------------------------
def train_one_fold(
    cfg: CFG,
    device: torch.device,
    df: pd.DataFrame,
    all_pids: List[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    event_data: Dict,
    results_tracker: Dict[str, Dict[str, List[float]]],
) -> float:
    """Train and evaluate MLP on one CV fold, return best val C-index."""
    skf_ratio = 1.0 / cfg.n_splits
    train_ids = [all_pids[i] for i in tr_idx]
    val_ids = [all_pids[i] for i in va_idx]

    # Verify split integrity and balance
    confirm_split_accuracy(
        patient_ids=all_pids,
        event_data=event_data,
        train_ids=train_ids,
        val_ids=val_ids,
        test_size=skf_ratio,
        tolerance=0.05,
    )
    key_numeric = [c for c in ["age", "no_instillations"] if c in df.columns]
    key_categorical = [
        c for c in ["sex", "smoking", "tumor", "stage", "substage", "grade", "reTUR", "LVI", "variant", "EORTC", "BRS"]
        if c in df.columns
    ]
    confirm_split_feature_balance(
        df=df,
        train_ids=train_ids,
        val_ids=val_ids,
        numeric_features=key_numeric,
        categorical_features=key_categorical,
    )

    # Split features/labels
    X_full = df.drop(columns=[cfg.survival_time_col, cfg.survival_event_col])
    y_full = df[[cfg.survival_event_col, cfg.survival_time_col]]
    X_tr, X_va = X_full.loc[train_ids], X_full.loc[val_ids]
    y_tr, y_va = y_full.loc[train_ids], y_full.loc[val_ids]

    # Preprocess (fit on train only)
    numerical = X_full.select_dtypes(include=np.number).columns.tolist()
    categorical = X_full.select_dtypes(exclude=np.number).columns.tolist()
    preproc = create_clinical_preprocessor(numerical, categorical)
    Xtr_proc = preproc.fit_transform(X_tr)
    Xva_proc = preproc.transform(X_va)

    # Datasets / loaders
    tr_ds = ClinicalDataset(Xtr_proc, y_tr[cfg.survival_event_col].values, y_tr[cfg.survival_time_col].values)
    va_ds = ClinicalDataset(Xva_proc, y_va[cfg.survival_event_col].values, y_va[cfg.survival_time_col].values)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model/optim
    feature_dim = Xtr_proc.shape[1]
    model = PredictionModel_Clinical(input_dim=feature_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    if cfg.wandb_watch:
        wandb.watch(model, log="all", log_freq=cfg.log_freq)
    optimzr = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Train and keep best epoch by val C-index
    best_c, best_state = -1.0, None
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            features = batch["features"].to(device)
            event = batch["event"].to(device)
            time = batch["time"].to(device)
            optimzr.zero_grad()
            risk = model(features).squeeze()
            loss = cox_loss(risk, event, time)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimzr.step()
            total_loss += float(loss.item())

        # Eval
        model.eval()
        val_scores, val_events, val_times = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                features = batch["features"].to(device)
                risk = model(features).detach().cpu().numpy().ravel()
                val_scores.extend(risk.tolist())
                val_events.extend(batch["event"].numpy())
                val_times.extend(batch["time"].numpy())
        c_idx = compute_cindex(np.array(val_events), np.array(val_times), np.array(val_scores))
        wandb.log({"avg_epoch_loss": total_loss / max(1, len(tr_loader)), "val_c_index": c_idx, "epoch": epoch + 1})

        if c_idx > best_c:
            best_c = float(c_idx)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # store per-fold train/val c-index
    with torch.no_grad():
        model.eval()
        tr_scores = model(torch.as_tensor(Xtr_proc, dtype=torch.float32).to(device)).cpu().numpy().ravel()
        va_scores = model(torch.as_tensor(Xva_proc, dtype=torch.float32).to(device)).cpu().numpy().ravel()
    tr_c = compute_cindex(y_tr[cfg.survival_event_col].values, y_tr[cfg.survival_time_col].values, tr_scores)
    va_c = compute_cindex(y_va[cfg.survival_event_col].values, y_va[cfg.survival_time_col].values, va_scores)
    results_tracker["MLP"]["train"].append(float(tr_c))
    results_tracker["MLP"]["test"].append(float(va_c))
    return best_c


def train_final_all_data(cfg: CFG, device: torch.device, df: pd.DataFrame, artifacts_dir: str) -> None:
    """Fit preprocessor on all data, train final model, and export artifacts."""
    X_full = df.drop(columns=[cfg.survival_time_col, cfg.survival_event_col])
    y_full = df[[cfg.survival_event_col, cfg.survival_time_col]]
    numerical = X_full.select_dtypes(include=np.number).columns.tolist()
    categorical = X_full.select_dtypes(exclude=np.number).columns.tolist()
    preproc = create_clinical_preprocessor(numerical, categorical)
    X_proc = preproc.fit_transform(X_full)

    model = PredictionModel_Clinical(input_dim=X_proc.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimzr = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ds = ClinicalDataset(X_proc, y_full[cfg.survival_event_col].values, y_full[cfg.survival_time_col].values)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            feats = batch["features"].to(device)
            event = batch["event"].to(device)
            time = batch["time"].to(device)
            optimzr.zero_grad()
            loss = cox_loss(model(feats).squeeze(), event, time)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimzr.step()
            total_loss += float(loss.item())
        wandb.log({"final_all_data_epoch": epoch + 1, "final_all_data_avg_loss": total_loss / max(1, len(loader))})

    # save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "clinical_mlp.pt"))
    joblib.dump(preproc, os.path.join(artifacts_dir, "clinical_preprocessor.joblib"))

    # portable JSON spec
    try:
        spec = {"version": 1, "numeric_cols": [], "categorical_cols": [], "numeric": {}, "categorical": {}}
        for name, pipe, cols in preproc.transformers_:
            if name == "num":
                spec["numeric_cols"] = list(cols)
                imputer = pipe.named_steps.get("imputer")
                scaler = pipe.named_steps.get("scaler")
                stats = {}
                if getattr(imputer, "statistics_", None) is not None:
                    for c, v in zip(cols, imputer.statistics_):
                        stats[c] = float(v) if v is not None and np.isfinite(v) else None
                mean_map, scale_map = {}, {}
                if getattr(scaler, "mean_", None) is not None:
                    for c, m, s in zip(cols, scaler.mean_, scaler.scale_):
                        mean_map[c] = float(m)
                        scale_map[c] = float(s) if s not in (0, None) and np.isfinite(s) else 1.0
                spec["numeric"]["imputer_statistics"] = stats
                spec["numeric"]["scaler_mean"] = mean_map
                spec["numeric"]["scaler_scale"] = scale_map
            elif name == "cat":
                spec["categorical_cols"] = list(cols)
                imputer = pipe.named_steps.get("imputer")
                onehot = pipe.named_steps.get("onehot")
                fill_map = {}
                if getattr(imputer, "statistics_", None) is not None:
                    for c, v in zip(cols, imputer.statistics_):
                        fill_map[c] = None if v is None else str(v)
                cats_map = {}
                if getattr(onehot, "categories_", None) is not None:
                    for c, cats in zip(cols, onehot.categories_):
                        cats_map[c] = [None if v is None else str(v) for v in list(cats)]
                spec["categorical"]["imputer_fill"] = fill_map
                spec["categorical"]["onehot_categories"] = cats_map
        with open(os.path.join(artifacts_dir, "clinical_preproc_spec.json"), "w") as f:
            json.dump(spec, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to export portable preprocessor spec: {e}")


# --------------------------
# Main
# --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/clinical_mlp.yaml")
    p.add_argument("--resources_dir", type=str, default=None, help="Override artifacts dir (optional)")
    return p.parse_args()


def main() -> None:
    import yaml

    args = parse_args()
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    cfg = CFG(**raw)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # init results + wandb
    results_dir = get_results_dir(cfg.results_scope)
    write_config_json(results_dir, asdict(cfg))
    wandb.init(project=cfg.project, config=asdict(cfg))

    # data & splits
    all_pids, event_data = load_clinical_data_for_stratification(cfg.data_dir)
    df = load_clinical_dataframe(cfg.data_dir, all_pids)
    splits = make_cv_splits(cfg, all_pids, df)

    # per-fold tracking
    agg: Dict[str, List[float]] = {"MLP": []}
    cindex_by_model: Dict[str, Dict[str, List[float]]] = {"MLP": {"train": [], "test": []}}

    # CV
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n========== Fold {fold_idx}/{cfg.n_splits} ==========")
        best_c = train_one_fold(cfg, device, df, all_pids, tr_idx, va_idx, event_data, cindex_by_model)
        agg["MLP"].append(best_c)
        wandb.log({"fold": fold_idx, "best_val_c_index": best_c})

    # summarize + save metrics
    def summarize(name: str, values: List[float]):
        if not values:
            return None
        return {
            "model": name,
            "c_index_mean": float(np.mean(values)),
            "c_index_min": float(np.min(values)),
            "c_index_max": float(np.max(values)),
        }

    summaries = []
    for k, v in agg.items():
        s = summarize(k, v)
        if s:
            summaries.append(s)
            print(f"{k}: mean={s['c_index_mean']:.4f}, min={s['c_index_min']:.4f}, max={s['c_index_max']:.4f}")
            append_result([], k, s["c_index_mean"], extra={"c_index_min": s["c_index_min"], "c_index_max": s["c_index_max"]})

    rows = [{"model": s["model"], "c_index": s["c_index_mean"], "c_index_min": s["c_index_min"], "c_index_max": s["c_index_max"]} for s in summaries]
    metrics_csv = write_metrics_csv(results_dir, rows)
    print(f"Saved metrics to: {metrics_csv}")

    # export per-fold + summary + config tables
    per_fold_rows = []
    for model_name, splits_dict in cindex_by_model.items():
        train_vals = splits_dict.get("train", [])
        test_vals = splits_dict.get("test", [])
        n = max(len(train_vals), len(test_vals))
        for i in range(n):
            tr_val = float(train_vals[i]) if i < len(train_vals) and np.isfinite(train_vals[i]) else float("nan")
            te_val = float(test_vals[i]) if i < len(test_vals) and np.isfinite(test_vals[i]) else float("nan")
            per_fold_rows.append({"fold": i + 1, "model": model_name, "train_c_index": tr_val, "test_c_index": te_val})
    pd.DataFrame(per_fold_rows).to_csv(os.path.join(results_dir, "per_fold_cindex.csv"), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(results_dir, "summary_cindex.csv"), index=False)
    pd.DataFrame(list(asdict(cfg).items()), columns=["param", "value"]).to_csv(os.path.join(results_dir, "config.csv"), index=False)

    # train final on all data and export artifacts
    repo_root = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(repo_root, cfg.resources_dir) if args.resources_dir is None else args.resources_dir
    train_final_all_data(cfg, device, df, artifacts_dir)

    print("\nTraining finished.")
    wandb.finish()


if __name__ == "__main__":
    main()
