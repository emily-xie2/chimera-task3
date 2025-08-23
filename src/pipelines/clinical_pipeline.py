import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import joblib

# Add the parent directory to Python path to find utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""MLP-only clinical pipeline using utilities under utils/"""

# Import shared components
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

# Note: Non-MLP baselines and visualizations removed for MLP-only run

if __name__ == "__main__":
    # 0. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    config = {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "num_epochs": 50,
        "model_hidden_dim": 64,
        "dropout": 0.4,
        "test_size": 0.2,
        "random_state": 42,
    }

    wandb.init(config=config, project="Clinical-Survival-Analysis-Stratified")

    # Results aggregation
    results_rows = []
    results_dir = get_results_dir("clinical")
    write_config_json(results_dir, dict(wandb.config))
    # Prepare package resources dir for exporting the best model weights
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_resources_dir = os.path.join(pkg_root, "/home/emxie/scratch/CHIMERA_minimal_baseline/chimera-task3-submission/resources")
    os.makedirs(pkg_resources_dir, exist_ok=True)

    # 1. Load and Preprocess Data
    data_dir = "data/"
    
    # Load clinical data for stratification
    all_patient_ids, event_data = load_clinical_data_for_stratification(data_dir)
    
    # Prepare 10-fold stratified CV using composite labels with rare-bin guarding
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=wandb.config.random_state)

    # Load clinical data into a single DataFrame once
    clinical_data_list = []
    for pid in all_patient_ids:
        cd_path = os.path.join(data_dir, pid, f"{pid}_CD.json")
        with open(cd_path, 'r') as f:
            data = json.load(f)
            data['pid'] = pid
            clinical_data_list.append(data)
    df = pd.DataFrame(clinical_data_list)
    df.set_index('pid', inplace=True)
    df.replace(-1, np.nan, inplace=True)
    
    # Define survival columns before any usage
    survival_time_col = 'Time_to_prog_or_FUend'
    survival_event_col = 'progression'

    # Build composite stratification labels after df is available
    strat_labels = build_composite_strat_labels(
        df=df,
        patient_ids=all_patient_ids,
        n_splits=skf.get_n_splits(),
        progression_col='progression',
        stage_col='stage',
        substage_col='substage',
        min_count=skf.get_n_splits(),
    )

    # Create folds with composite labels
    all_folds = list(skf.split(all_patient_ids, strat_labels))

    X_full = df.drop(columns=[survival_time_col, survival_event_col])
    y_full = df[[survival_event_col, survival_time_col]]

    numerical_features = X_full.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_full.select_dtypes(exclude=np.number).columns.tolist()

    # Aggregators across folds (MLP only)
    agg = {
        "MLP": [],
    }
    # Track train/test C-index per fold (MLP only)
    cindex_by_model = {
        "MLP": {"train": [], "test": []},
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(all_folds, start=1):
        print(f"\n========== Fold {fold_idx}/{skf.get_n_splits()} ==========")
        train_ids = [all_patient_ids[i] for i in tr_idx]
        val_ids = [all_patient_ids[i] for i in va_idx]
        # Verify split integrity and stratification within each fold
        confirm_split_accuracy(
            patient_ids=all_patient_ids,
            event_data=event_data,
            train_ids=train_ids,
            val_ids=val_ids,
            test_size=1.0 / skf.get_n_splits(),
            tolerance=0.05,
        )
        # Check that key clinical features are balanced between train/val
        key_numeric = [c for c in ['age', 'no_instillations'] if c in df.columns]
        key_categorical = [
            c for c in [
                'sex', 'smoking', 'tumor', 'stage', 'substage', 'grade', 'reTUR',
                'LVI', 'variant', 'EORTC', 'BRS'
            ] if c in df.columns
        ]
        confirm_split_feature_balance(
            df=df,
            train_ids=train_ids,
            val_ids=val_ids,
            numeric_features=key_numeric,
            categorical_features=key_categorical,
        )

        # Preprocessing per fold (fit on train only)
        preprocessor = create_clinical_preprocessor(numerical_features, categorical_features)
        X_train = X_full.loc[train_ids]
        X_val = X_full.loc[val_ids]
        y_train = y_full.loc[train_ids]
        y_val = y_full.loc[val_ids]
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        # Fold tag
        fold_tag  = f"Fold {fold_idx}"


        # Datasets and loaders
        train_dataset = ClinicalDataset(
            X_train_processed,
            y_train[survival_event_col].values,
            y_train[survival_time_col].values,
        )
        val_dataset = ClinicalDataset(
            X_val_processed,
            y_val[survival_event_col].values,
            y_val[survival_time_col].values,
        )
        train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

        # Model
        feature_dim = X_train_processed.shape[1]
        wandb.config.update({"feature_dim": feature_dim})
        model = PredictionModel_Clinical(
            input_dim=feature_dim,
            hidden_dim=wandb.config.model_hidden_dim,
            dropout=wandb.config.dropout,
        ).to(device)
        print(f"Initialized clinical model with input dimension: {feature_dim}")
        wandb.watch(model, log="all", log_freq=5)

        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

        print("Training Clinical MLP...")
        last_c_index = 0.5
        fold_best_c = -1.0
        fold_best_state = None
        for epoch in range(wandb.config.num_epochs):
            model.train()
            total_train_loss = 0.0
            for batch in train_loader:
                features = batch['features'].to(device)
                event = batch['event'].to(device)
                time = batch['time'].to(device)
                optimizer.zero_grad()
                risk_scores = model(features)
                loss = cox_loss(risk_scores.squeeze(), event, time)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Eval
            model.eval()
            val_risk_scores, val_events, val_times = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    risk_scores = model(features)
                    arr = risk_scores.detach().cpu().numpy()
                    val_risk_scores.extend(np.atleast_1d(arr).ravel().tolist())
                    val_events.extend(batch['event'].numpy())
                    val_times.extend(batch['time'].numpy())
            last_c_index = compute_cindex(val_events, val_times, val_risk_scores)
            # Track best checkpoint for this fold (in-memory only)
            if last_c_index > fold_best_c:
                fold_best_c = float(last_c_index)
                fold_best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            print(
                f"Fold {fold_idx} | Epoch [{epoch+1}/{wandb.config.num_epochs}] | "
                f"Avg Train Loss: {avg_train_loss:.4f} | Val C-Index: {last_c_index:.4f}"
            )
            wandb.log({
                "fold": fold_idx,
                "epoch": epoch + 1,
                "avg_epoch_loss": avg_train_loss,
                "c_index": last_c_index,
            })

        # (Removed per request) No feature distribution plots

        # Ensure we use the best-epoch weights for predictions on this fold
        if fold_best_state is not None:
            model.load_state_dict(fold_best_state)

        agg["MLP"].append(float(fold_best_c))

        # Record per-fold MLP train/test C-index (MLP-only pipeline)
        try:
            y_train_df = pd.DataFrame({
                'time': y_train[survival_time_col].values,
                'event': y_train[survival_event_col].values,
            })
            y_val_df = pd.DataFrame({
                'time': y_val[survival_time_col].values,
                'event': y_val[survival_event_col].values,
            })
            with torch.no_grad():
                model.eval()
                tr_logits = model(torch.as_tensor(X_train_processed, dtype=torch.float32).to(device))
                va_logits = model(torch.as_tensor(X_val_processed, dtype=torch.float32).to(device))
                tr_scores = np.atleast_1d(tr_logits.detach().cpu().numpy()).ravel()
                va_scores = np.atleast_1d(va_logits.detach().cpu().numpy()).ravel()
            mlp_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, tr_scores)
            mlp_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, va_scores)
            cindex_by_model["MLP"]["train"].append(float(mlp_tr_c))
            cindex_by_model["MLP"]["test"].append(float(mlp_te_c))
        except Exception as _e:
            print(f"Warning: failed to compute per-fold MLP c-indices on fold {fold_idx}: {_e}")

        # Skip non-MLP baselines and ensembling
        continue

        # =============================
        # Ensembling on this fold
        # =============================
        try:
            # Collect base model risk predictions for both train and val
            base_val_preds = {}
            base_tr_preds = {}
            fold_val_cis = {}

            # 1) Clinical MLP predictions (already have val; also compute train)
            with torch.no_grad():
                model.eval()
                tr_logits = model(torch.as_tensor(X_train_processed, dtype=torch.float32).to(device))
                va_logits = model(torch.as_tensor(X_val_processed, dtype=torch.float32).to(device))
                base_tr_preds['mlp'] = np.atleast_1d(tr_logits.detach().cpu().numpy()).ravel()
                base_val_preds['mlp'] = np.atleast_1d(va_logits.detach().cpu().numpy()).ravel()
            # Record train/test C-index for MLP
            try:
                mlp_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['mlp'])
                mlp_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['mlp'])
                fold_val_cis['mlp'] = float(mlp_te_c)
                cindex_by_model["MLP"]["train"].append(float(mlp_tr_c))
                cindex_by_model["MLP"]["test"].append(float(mlp_te_c))
            except Exception as _e:
                print(f"MLP C-index logging failed on fold {fold_idx}: {_e}")

            # 2) CoxPH (fit with scaling)
            try:
                scaler_cox = StandardScaler()
                Xtr_scaled = scaler_cox.fit_transform(X_train_processed)
                Xva_scaled = scaler_cox.transform(X_val_processed)
                from sksurv.linear_model import CoxPHSurvivalAnalysis
                ytr_struct = np.array(list(zip(y_train_df['event'].astype(bool).values, y_train_df['time'].values)), dtype=[('event', bool), ('time', float)])
                cox = CoxPHSurvivalAnalysis(alpha=0.01, ties='efron', n_iter=100)
                cox.fit(Xtr_scaled, ytr_struct)
                base_tr_preds['coxph'] = cox.predict(Xtr_scaled)
                base_val_preds['coxph'] = cox.predict(Xva_scaled)
                # Record train/test C-index for CoxPH
                try:
                    cox_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['coxph'])
                    cox_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['coxph'])
                    fold_val_cis['coxph'] = float(cox_te_c)
                    cindex_by_model["CoxPH"]["train"].append(float(cox_tr_c))
                    cindex_by_model["CoxPH"]["test"].append(float(cox_te_c))
                except Exception as __e:
                    print(f"CoxPH C-index logging failed on fold {fold_idx}: {__e}")
            except Exception as _e:
                print(f"CoxPH predictions failed on fold {fold_idx}: {_e}")

            # 3) RSF
            try:
                from sksurv.ensemble import RandomSurvivalForest
                ytr_struct = np.array(list(zip(y_train_df['event'].astype(bool).values, y_train_df['time'].values)), dtype=[('event', bool), ('time', float)])
                rsf = RandomSurvivalForest(n_estimators=50, random_state=42)
                rsf.fit(X_train_processed, ytr_struct)
                base_tr_preds['rsf'] = rsf.predict(X_train_processed)
                base_val_preds['rsf'] = rsf.predict(X_val_processed)
                # Record train/test C-index for RSF
                try:
                    rsf_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['rsf'])
                    rsf_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['rsf'])
                    fold_val_cis['rsf'] = float(rsf_te_c)
                    cindex_by_model["RSF"]["train"].append(float(rsf_tr_c))
                    cindex_by_model["RSF"]["test"].append(float(rsf_te_c))
                except Exception as __e:
                    print(f"RSF C-index logging failed on fold {fold_idx}: {__e}")
            except Exception as _e:
                print(f"RSF predictions failed on fold {fold_idx}: {_e}")

            # 4) RealMLP
            try:
                rmlp = RealMLPSurvival(num_bins=11, device=str(device))
                # Handle censored training labels as in trainer
                y_train_corr = y_train_df.copy()
                max_observed_time = y_train_corr['time'].max()
                y_train_corr.loc[y_train_corr['event'] == 0, 'time'] = max_observed_time + 1
                rmlp.fit(X_train_processed, y_train_corr)
                base_tr_preds['realmlp'] = rmlp.predict_risk_score(X_train_processed)
                base_val_preds['realmlp'] = rmlp.predict_risk_score(X_val_processed)
                # Record train/test C-index for RealMLP
                try:
                    rmlp_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['realmlp'])
                    rmlp_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['realmlp'])
                    fold_val_cis['realmlp'] = float(rmlp_te_c)
                    cindex_by_model["RealMLP"]["train"].append(float(rmlp_tr_c))
                    cindex_by_model["RealMLP"]["test"].append(float(rmlp_te_c))
                except Exception as __e:
                    print(f"RealMLP C-index logging failed on fold {fold_idx}: {__e}")
            except Exception as _e:
                print(f"RealMLP predictions failed on fold {fold_idx}: {_e}")

            # 5) TabM (use DataFrame interface)
            try:
                Xtr_df = pd.DataFrame(X_train_processed)
                Xva_df = pd.DataFrame(X_val_processed)
                tabm = TabMSurvival(num_bins=11, k=16, device=str(device), epochs=50, lr=1e-3)
                tabm.fit(Xtr_df, y_train_df)
                base_tr_preds['tabm'] = tabm.predict_risk_score(Xtr_df)
                base_val_preds['tabm'] = tabm.predict_risk_score(Xva_df)
                # Record train/test C-index for TabM
                try:
                    tabm_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['tabm'])
                    tabm_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['tabm'])
                    fold_val_cis['tabm'] = float(tabm_te_c)
                    cindex_by_model["TabM"]["train"].append(float(tabm_tr_c))
                    cindex_by_model["TabM"]["test"].append(float(tabm_te_c))
                except Exception as __e:
                    print(f"TabM C-index logging failed on fold {fold_idx}: {__e}")
            except Exception as _e:
                print(f"TabM predictions failed on fold {fold_idx}: {_e}")

            # 6) XGBSE
            try:
                try:
                    from xgbse import XGBSEKaplanNeighbors
                except Exception as __imp_e:
                    raise __imp_e
                Xtr_np = np.array(X_train_processed, dtype=np.float64)
                Xva_np = np.array(X_val_processed, dtype=np.float64)
                Xtr_np = np.nan_to_num(Xtr_np, nan=0.0)
                Xva_np = np.nan_to_num(Xva_np, nan=0.0)
                ytr_struct = np.array(list(zip(y_train_df['event'].astype(bool).values, y_train_df['time'].values)), dtype=[('event', bool), ('time', float)])
                xgb_model = XGBSEKaplanNeighbors(n_neighbors=30)
                xgb_model.fit(Xtr_np, ytr_struct)
                pred_tr = xgb_model.predict(Xtr_np)
                pred_va = xgb_model.predict(Xva_np)
                if pred_tr.ndim > 1:
                    risk_tr = 1 - np.mean(pred_tr, axis=1)
                else:
                    risk_tr = 1 - pred_tr
                if pred_va.ndim > 1:
                    risk_va = 1 - np.mean(pred_va, axis=1)
                else:
                    risk_va = 1 - pred_va
                base_tr_preds['xgbse'] = np.asarray(risk_tr).reshape(-1)
                base_val_preds['xgbse'] = np.asarray(risk_va).reshape(-1)
                # Record train/test C-index for XGBSE
                try:
                    xgb_tr_c = compute_cindex(y_train_df['event'].values, y_train_df['time'].values, base_tr_preds['xgbse'])
                    xgb_te_c = compute_cindex(y_val_df['event'].values, y_val_df['time'].values, base_val_preds['xgbse'])
                    fold_val_cis['xgbse'] = float(xgb_te_c)
                    cindex_by_model["XGBSE"]["train"].append(float(xgb_tr_c))
                    cindex_by_model["XGBSE"]["test"].append(float(xgb_te_c))
                except Exception as __e:
                    print(f"XGBSE C-index logging failed on fold {fold_idx}: {__e}")
            except Exception as _e:
                print(f"XGBSE predictions failed on fold {fold_idx}: {_e}")

            # Removed DeepCox and AutoDSM per request

            # Helper: z-score
            def _z(x):
                x = np.asarray(x, dtype=float)
                return (x - np.mean(x)) / (np.std(x) + 1e-8)

            # Establish model order present
            model_keys = list(base_val_preds.keys())
            if len(model_keys) >= 2:
                # Prepare matrices
                P_va = np.column_stack([_z(base_val_preds[k]) for k in model_keys])
                # Align train columns with available keys
                aligned_keys = [k for k in model_keys if k in base_tr_preds]
                P_tr = None
                if len(aligned_keys) >= 2:
                    P_tr = np.column_stack([_z(base_tr_preds[k]) for k in aligned_keys])

                val_events = y_val_df['event'].astype(bool).values
                val_times = y_val_df['time'].values
                tr_events = y_train_df['event'].astype(bool).values
                tr_times = y_train_df['time'].values

                # MLP+RSF ensemble (z-score average of portable models)
                try:
                    if 'mlp' in model_keys and 'rsf' in model_keys:
                        ens_keys = ['mlp', 'rsf']
                        P_va_mr = np.column_stack([_z(base_val_preds[k]) for k in ens_keys])
                        ens_mr = np.mean(P_va_mr, axis=1)
                        c_mr = compute_cindex(val_events, val_times, ens_mr)
                        agg["MLP_RSF"].append(float(c_mr))
                        print(f"Fold {fold_idx} | Ensemble MLP+RSF C-Index: {c_mr:.4f}")
                        # Track best bundle with per-model mean/std
                        try:
                            if float(c_mr) > best_mlp_rsf_c:
                                stats = {}
                                for k in ens_keys:
                                    vec = np.asarray(base_val_preds[k], dtype=float)
                                    mu = float(np.mean(vec)) if vec.size > 0 else 0.0
                                    sd = float(np.std(vec)) if vec.size > 0 else 1.0
                                    if not np.isfinite(sd) or sd < 1e-8:
                                        sd = 1.0
                                    stats[k] = (mu, sd)
                                best_mlp_rsf_c = float(c_mr)
                                best_mlp_rsf_bundle = {
                                    "aligned_keys": ens_keys,
                                    "stats": stats,
                                }
                        except Exception as __ie:
                            print(f"Warning: failed to update best MLP+RSF bundle: {__ie}")
                        if P_tr is not None and all(k in aligned_keys for k in ens_keys):
                            P_tr_mr = np.column_stack([_z(base_tr_preds[k]) for k in ens_keys])
                            ens_mr_tr = np.mean(P_tr_mr, axis=1)
                            c_mr_tr = compute_cindex(tr_events, tr_times, ens_mr_tr)
                            cindex_by_model["MLP_RSF"]["train"].append(float(c_mr_tr))
                        else:
                            cindex_by_model["MLP_RSF"]["train"].append(float('nan'))
                        cindex_by_model["MLP_RSF"]["test"].append(float(c_mr))
                    else:
                        agg["MLP_RSF"].append(float('nan'))
                        cindex_by_model["MLP_RSF"]["train"].append(float('nan'))
                        cindex_by_model["MLP_RSF"]["test"].append(float('nan'))
                except Exception as __e:
                    print(f"MLP+RSF ensemble failed on fold {fold_idx}: {__e}")

                # Load optional weights from a prior metrics.csv if available
                weights = None
                try:
                    possible_metrics_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/449076/metrics.csv")
                    if os.path.exists(possible_metrics_csv):
                        mdf = pd.read_csv(possible_metrics_csv)
                        name_to_key = {
                            'MLP': 'mlp',
                            'CoxPH': 'coxph',
                            'RSF': 'rsf',
                            'RealMLP': 'realmlp',
                            'TabM': 'tabm',
                            'XGBSE': 'xgbse',
                        }
                        w_map = {}
                        for _, row in mdf.iterrows():
                            if row['model'] in name_to_key:
                                w_map[name_to_key[row['model']]] = float(row['c_index'])
                        w = np.array([w_map.get(k, 0.0) for k in model_keys], dtype=float)
                        if np.sum(w) > 0:
                            weights = w / np.sum(w)
                except Exception as _e:
                    print(f"Could not load ensemble weights from metrics.csv: {_e}")

                # Fallback to uniform weights
                if weights is None:
                    weights = np.ones(len(model_keys), dtype=float) / len(model_keys)

                # Weighted z-score average
                ens_weighted = P_va @ weights
                c_w = compute_cindex(val_events, val_times, ens_weighted)
                agg["Weighted"].append(float(c_w))
                print(f"Fold {fold_idx} | Ensemble Weighted C-Index: {c_w:.4f}")
                # Train side (if we have aligned train preds)
                if P_tr is not None:
                    # Map weights to aligned keys
                    idxs = [model_keys.index(k) for k in aligned_keys]
                    w_sub = weights[idxs]
                    if np.sum(w_sub) > 0:
                        w_sub = w_sub / np.sum(w_sub)
                    ens_weighted_tr = P_tr @ w_sub
                    c_w_tr = compute_cindex(tr_events, tr_times, ens_weighted_tr)
                    cindex_by_model["Weighted"]["train"].append(float(c_w_tr))
                else:
                    cindex_by_model["Weighted"]["train"].append(float('nan'))
                cindex_by_model["Weighted"]["test"].append(float(c_w))

                # Simple mean of z-scores (uniform average)
                ens_mean = np.mean(P_va, axis=1)
                c_mean = compute_cindex(val_events, val_times, ens_mean)
                agg["Mean"].append(float(c_mean))
                print(f"Fold {fold_idx} | Ensemble Mean C-Index: {c_mean:.4f}")
                if P_tr is not None:
                    ens_mean_tr = np.mean(P_tr, axis=1)
                    c_mean_tr = compute_cindex(tr_events, tr_times, ens_mean_tr)
                    cindex_by_model["Mean"]["train"].append(float(c_mean_tr))
                else:
                    cindex_by_model["Mean"]["train"].append(float('nan'))
                cindex_by_model["Mean"]["test"].append(float(c_mean))

                # Rank-average
                ranks = []
                for j in range(P_va.shape[1]):
                    col = P_va[:, j]
                    order = np.argsort(col)
                    r = np.empty_like(order)
                    r[order] = np.arange(len(col))
                    ranks.append(r.astype(float))
                ranks = np.column_stack(ranks)
                ens_rank = np.mean(ranks, axis=1)
                c_r = compute_cindex(val_events, val_times, ens_rank)
                agg["RankAvg"].append(float(c_r))
                print(f"Fold {fold_idx} | Ensemble RankAvg C-Index: {c_r:.4f}")
                # Train side ranks
                if P_tr is not None:
                    ranks_tr = []
                    for j in range(P_tr.shape[1]):
                        col = P_tr[:, j]
                        order = np.argsort(col)
                        r = np.empty_like(order)
                        r[order] = np.arange(len(col))
                        ranks_tr.append(r.astype(float))
                    ranks_tr = np.column_stack(ranks_tr)
                    ens_rank_tr = np.mean(ranks_tr, axis=1)
                    c_r_tr = compute_cindex(tr_events, tr_times, ens_rank_tr)
                    cindex_by_model["RankAvg"]["train"].append(float(c_r_tr))
                else:
                    cindex_by_model["RankAvg"]["train"].append(float('nan'))
                cindex_by_model["RankAvg"]["test"].append(float(c_r))

                # Weighted rank-average (using same weights if provided)
                try:
                    # weights correspond to model_keys; compute weighted mean of ranks
                    wranks = ranks * weights[np.newaxis, :]
                    ens_wrank = np.sum(wranks, axis=1) / (np.sum(weights) + 1e-12)
                    c_wr = compute_cindex(val_events, val_times, ens_wrank)
                    agg["WeightedRankAvg"].append(float(c_wr))
                    print(f"Fold {fold_idx} | Ensemble WeightedRankAvg C-Index: {c_wr:.4f}")
                    if P_tr is not None:
                        # Train side weighted ranks
                        ranks_tr = []
                        for j in range(P_tr.shape[1]):
                            col = P_tr[:, j]
                            order = np.argsort(col)
                            r = np.empty_like(order)
                            r[order] = np.arange(len(col))
                            ranks_tr.append(r.astype(float))
                        ranks_tr = np.column_stack(ranks_tr)
                        # Map weights to aligned keys
                        idxs = [model_keys.index(k) for k in aligned_keys]
                        w_sub = weights[idxs]
                        wranks_tr = ranks_tr * w_sub[np.newaxis, :]
                        ens_wrank_tr = np.sum(wranks_tr, axis=1) / (np.sum(w_sub) + 1e-12)
                        c_wr_tr = compute_cindex(tr_events, tr_times, ens_wrank_tr)
                        cindex_by_model["WeightedRankAvg"]["train"].append(float(c_wr_tr))
                    else:
                        cindex_by_model["WeightedRankAvg"]["train"].append(float('nan'))
                    cindex_by_model["WeightedRankAvg"]["test"].append(float(c_wr))
                except Exception as __e:
                    print(f"WeightedRankAvg failed on fold {fold_idx}: {__e}")

                # Median of z-scores
                ens_median = np.median(P_va, axis=1)
                c_m = compute_cindex(val_events, val_times, ens_median)
                agg["Median"].append(float(c_m))
                print(f"Fold {fold_idx} | Ensemble Median C-Index: {c_m:.4f}")
                if P_tr is not None:
                    ens_median_tr = np.median(P_tr, axis=1)
                    c_m_tr = compute_cindex(tr_events, tr_times, ens_median_tr)
                    cindex_by_model["Median"]["train"].append(float(c_m_tr))
                else:
                    cindex_by_model["Median"]["train"].append(float('nan'))
                cindex_by_model["Median"]["test"].append(float(c_m))

                # Trimmed mean of z-scores (drop 1 lowest and 1 highest if >=3 models)
                try:
                    if P_va.shape[1] >= 3:
                        sorted_va = np.sort(P_va, axis=1)
                        ens_tmean = np.mean(sorted_va[:, 1:-1], axis=1)
                    else:
                        ens_tmean = np.mean(P_va, axis=1)
                    c_tm = compute_cindex(val_events, val_times, ens_tmean)
                    agg["TrimmedMean"].append(float(c_tm))
                    print(f"Fold {fold_idx} | Ensemble TrimmedMean C-Index: {c_tm:.4f}")
                    if P_tr is not None:
                        if P_tr.shape[1] >= 3:
                            sorted_tr = np.sort(P_tr, axis=1)
                            ens_tmean_tr = np.mean(sorted_tr[:, 1:-1], axis=1)
                        else:
                            ens_tmean_tr = np.mean(P_tr, axis=1)
                        c_tm_tr = compute_cindex(tr_events, tr_times, ens_tmean_tr)
                        cindex_by_model["TrimmedMean"]["train"].append(float(c_tm_tr))
                    else:
                        cindex_by_model["TrimmedMean"]["train"].append(float('nan'))
                    cindex_by_model["TrimmedMean"]["test"].append(float(c_tm))
                except Exception as __e:
                    print(f"TrimmedMean failed on fold {fold_idx}: {__e}")

                # Top-K mean (K=3) of z-scores by current fold validation C-index
                try:
                    if len(fold_val_cis) > 0:
                        # Restrict TopKMean to portable keys available at inference (MLP, CoxPH, RSF)
                        portable_candidates = ['mlp', 'coxph', 'rsf']
                        portable_keys = [k for k in portable_candidates if k in model_keys and k in fold_val_cis and np.isfinite(fold_val_cis[k])]
                        if len(portable_keys) < 2:
                            raise RuntimeError("TopKMean requires at least two portable base models (e.g., 'mlp' and 'coxph')")
                        P_va_top = np.column_stack([_z(base_val_preds[k]) for k in portable_keys])
                        ens_topk = np.mean(P_va_top, axis=1)
                        c_topk = compute_cindex(val_events, val_times, ens_topk)
                        agg["TopKMean"].append(float(c_topk))
                        print(f"Fold {fold_idx} | Ensemble TopKMean (portable: {portable_keys}) C-Index: {c_topk:.4f}")
                        if P_tr is not None:
                            aligned_top_keys = [k for k in portable_keys if k in aligned_keys]
                            if len(aligned_top_keys) >= 2:
                                P_tr_top = np.column_stack([_z(base_tr_preds[k]) for k in aligned_top_keys])
                                ens_topk_tr = np.mean(P_tr_top, axis=1)
                                c_topk_tr = compute_cindex(tr_events, tr_times, ens_topk_tr)
                                cindex_by_model["TopKMean"]["train"].append(float(c_topk_tr))
                            else:
                                cindex_by_model["TopKMean"]["train"].append(float('nan'))
                        else:
                            cindex_by_model["TopKMean"]["train"].append(float('nan'))
                        cindex_by_model["TopKMean"]["test"].append(float(c_topk))

                        # Keep best TopKMean bundle for export (store per-model mean/std from validation preds)
                        if float(c_topk) > best_topk_c:
                            stats = {}
                            for k in portable_keys:
                                vec = np.asarray(base_val_preds[k], dtype=float)
                                mu = float(np.mean(vec)) if vec.size > 0 else 0.0
                                sd = float(np.std(vec)) if vec.size > 0 else 1.0
                                if not np.isfinite(sd) or sd < 1e-8:
                                    sd = 1.0
                                stats[k] = (mu, sd)
                            best_topk_c = float(c_topk)
                            best_topk_bundle = {
                                "aligned_keys": portable_keys,
                                "stats": stats,  # key -> (mean, std) on validation preds
                            }
                    else:
                        agg["TopKMean"].append(float('nan'))
                        cindex_by_model["TopKMean"]["train"].append(float('nan'))
                        cindex_by_model["TopKMean"]["test"].append(float('nan'))
                except Exception as __e:
                    print(f"TopKMean failed on fold {fold_idx}: {__e}")

                # Cox stacking over top-3 base models by validation C-index
                try:
                    if P_tr is not None:
                        from sksurv.linear_model import CoxPHSurvivalAnalysis
                        scaler_meta = StandardScaler()
                        # pick top 3 models by current fold validation C-index
                        sorted_models = sorted(
                            [(k, v) for k, v in fold_val_cis.items() if k in aligned_keys and np.isfinite(v)],
                            key=lambda kv: -kv[1]
                        )
                        top_keys = [k for k, _ in sorted_models[:3]]
                        if len(top_keys) < 2:
                            top_keys = aligned_keys
                        P_tr_top = np.column_stack([_z(base_tr_preds[k]) for k in top_keys])
                        P_va_top = np.column_stack([_z(base_val_preds[k]) for k in top_keys])
                        P_tr_s = scaler_meta.fit_transform(P_tr_top)
                        P_va_s = scaler_meta.transform(P_va_top)
                        ytr_struct = np.array(list(zip(y_train_df['event'].astype(bool).values, y_train_df['time'].values)), dtype=[('event', bool), ('time', float)])
                        meta = CoxPHSurvivalAnalysis(alpha=0.01, ties='efron', n_iter=200)
                        meta.fit(P_tr_s, ytr_struct)
                        ens_stack = meta.predict(P_va_s)
                        c_s = compute_cindex(val_events, val_times, ens_stack)
                        agg["CoxStack"].append(float(c_s))
                        print(f"Fold {fold_idx} | Ensemble CoxStack (top-3) C-Index: {c_s:.4f}")
                        # Train c-index (in-sample)
                        ens_stack_tr = meta.predict(P_tr_s)
                        c_s_tr = compute_cindex(tr_events, tr_times, ens_stack_tr)
                        cindex_by_model["CoxStack"]["train"].append(float(c_s_tr))
                        cindex_by_model["CoxStack"]["test"].append(float(c_s))
                        # Keep best model bundle for export
                        if float(c_s) > best_coxstack_c:
                            best_coxstack_c = float(c_s)
                            best_coxstack_bundle = {
                                "meta": meta,
                                "scaler": scaler_meta,
                                "aligned_keys": top_keys,
                            }
                    else:
                        # No train predictions available to fit meta-learner
                        cindex_by_model["CoxStack"]["train"].append(float('nan'))
                        cindex_by_model["CoxStack"]["test"].append(float('nan'))
                except Exception as _e:
                    print(f"CoxStack failed on fold {fold_idx}: {_e}")
        except Exception as e:
            print(f"Warning: ensembling failed on fold {fold_idx}: {e}")

    # Aggregate across folds
    def summarize(name, values):
        if len(values) == 0:
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
        if s is not None:
            summaries.append(s)
            print(f"{k}: mean={s['c_index_mean']:.4f}, min={s['c_index_min']:.4f}, max={s['c_index_max']:.4f}")
            append_result(results_rows, k, s['c_index_mean'], extra={
                "c_index_min": s['c_index_min'],
                "c_index_max": s['c_index_max'],
            })

    # Persist metrics
    metrics_csv = write_metrics_csv(results_dir, results_rows)
    print(f"Saved metrics to: {metrics_csv}")

    # MLP-only: skip ensemble artifact saving and plotting

    # Build and save organized spreadsheets with key metrics
    try:
        # Per-fold long-form table
        per_fold_rows = []
        try:
            num_folds = skf.get_n_splits()
        except Exception:
            num_folds = None
        for model_name, splits in cindex_by_model.items():
            train_vals = splits.get('train', [])
            test_vals = splits.get('test', [])
            n = max(len(train_vals), len(test_vals))
            for i in range(n):
                tr_val = float(train_vals[i]) if i < len(train_vals) and np.isfinite(train_vals[i]) else float('nan')
                te_val = float(test_vals[i]) if i < len(test_vals) and np.isfinite(test_vals[i]) else float('nan')
                per_fold_rows.append({
                    'fold': i + 1,
                    'model': model_name,
                    'train_c_index': tr_val,
                    'test_c_index': te_val,
                })
        df_per_fold = pd.DataFrame(per_fold_rows)

        # Summary table from previously computed summaries
        if 'summaries' in locals() and len(summaries) > 0:
            df_summary = pd.DataFrame(summaries)
        else:
            df_summary = pd.DataFrame(columns=['model', 'c_index_mean', 'c_index_min', 'c_index_max'])

        # Config table
        df_config = pd.DataFrame(list(dict(wandb.config).items()), columns=['param', 'value'])

        # Write CSVs
        per_fold_csv = os.path.join(results_dir, 'per_fold_cindex.csv')
        summary_csv = os.path.join(results_dir, 'summary_cindex.csv')
        config_csv = os.path.join(results_dir, 'config.csv')
        df_per_fold.to_csv(per_fold_csv, index=False)
        df_summary.to_csv(summary_csv, index=False)
        df_config.to_csv(config_csv, index=False)
        print(f"Saved per-fold metrics to: {per_fold_csv}")
        print(f"Saved summary metrics to: {summary_csv}")
        print(f"Saved run config to: {config_csv}")

        # Extended summary metrics (mean, std, min, quartiles, max, count) per model and split
        try:
            stats_rows = []

            def _compute_stats(values):
                series = pd.Series(values, dtype=float)
                series = series[np.isfinite(series)]
                if series.empty:
                    return None
                return {
                    'mean': float(series.mean()),
                    'std': float(series.std(ddof=1)) if series.size > 1 else 0.0,
                    'min': float(series.min()),
                    '25%': float(series.quantile(0.25)),
                    '50%': float(series.median()),
                    '75%': float(series.quantile(0.75)),
                    'max': float(series.max()),
                    'n': int(series.size),
                }

            for model_name, splits in cindex_by_model.items():
                for split_name in ['train', 'test']:
                    vals = splits.get(split_name, [])
                    stats = _compute_stats([v for v in vals if np.isfinite(v)])
                    if stats is None:
                        continue
                    row = {'model': model_name, 'split': split_name}
                    row.update(stats)
                    stats_rows.append(row)

            df_extended = pd.DataFrame(stats_rows)
            extended_csv = os.path.join(results_dir, 'summary_cindex_extended.csv')
            df_extended.to_csv(extended_csv, index=False)
            print(f"Saved detailed summary metrics to: {extended_csv}")
        except Exception as _e:
            print(f"Warning: failed to save detailed summary metrics: {_e}")

        # Write Excel workbook (optional; falls back to CSVs if engine missing)
        excel_path = os.path.join(results_dir, 'results_summary.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df_summary.to_excel(writer, index=False, sheet_name='Summary')
                df_per_fold.to_excel(writer, index=False, sheet_name='PerFold')
                df_config.to_excel(writer, index=False, sheet_name='Config')
            print(f"Saved organized spreadsheet to: {excel_path}")
        except Exception as _e:
            print(f"Note: Excel export unavailable ({_e}); CSVs were saved instead.")
    except Exception as e:
        print(f"Warning: failed to save organized spreadsheets: {e}")

    # Train final model on ALL available data for inference-time weights
    try:
        print("\nTraining final clinical MLP on all data for inference weights...")
        full_preprocessor = create_clinical_preprocessor(numerical_features, categorical_features)
        X_full_processed = full_preprocessor.fit_transform(X_full)
        # Sanity check: ensure we are training on ALL available patients
        try:
            n_patients = len(df)
            print(f"Final training uses all patients: {X_full_processed.shape[0]} of {n_patients}")
            assert X_full_processed.shape[0] == n_patients
        except Exception as _e:
            print(f"Note: could not confirm full patient count exactly ({_e})")
        full_dataset = ClinicalDataset(
            X_full_processed,
            y_full[survival_event_col].values,
            y_full[survival_time_col].values,
        )
        full_loader = DataLoader(full_dataset, batch_size=wandb.config.batch_size, shuffle=True)

        full_feature_dim = X_full_processed.shape[1]
        final_model = PredictionModel_Clinical(
            input_dim=full_feature_dim,
            hidden_dim=wandb.config.model_hidden_dim,
            dropout=wandb.config.dropout,
        ).to(device)
        final_optimizer = optim.Adam(final_model.parameters(), lr=wandb.config.learning_rate)

        for epoch in range(wandb.config.num_epochs):
            final_model.train()
            total_loss = 0.0
            for batch in full_loader:
                features = batch['features'].to(device)
                event = batch['event'].to(device)
                time = batch['time'].to(device)
                final_optimizer.zero_grad()
                risk_scores = final_model(features)
                loss = cox_loss(risk_scores.squeeze(), event, time)
                if not torch.isnan(loss):
                    loss.backward()
                    final_optimizer.step()
                    total_loss += loss.item()
            avg_loss = total_loss / max(1, len(full_loader))
            print(f"Final (all-data) training | Epoch [{epoch+1}/{wandb.config.num_epochs}] | Avg Loss: {avg_loss:.4f}")
            wandb.log({
                "final_all_data_epoch": epoch + 1,
                "final_all_data_avg_loss": avg_loss,
            })

        # Save the full-data trained model weights and fitted preprocessor
        dst = os.path.join(pkg_resources_dir, "clinical_mlp.pt")
        torch.save(final_model.state_dict(), dst)
        preproc_dst = os.path.join(pkg_resources_dir, "clinical_preprocessor.joblib")
        joblib.dump(full_preprocessor, preproc_dst)
        # Also export a portable JSON spec for cross-version inference
        try:
            spec = {"version": 1}
            spec["numeric_cols"], spec["categorical_cols"] = [], []
            spec["numeric"], spec["categorical"] = {}, {}
            # Extract components from ColumnTransformer
            for name, pipe, cols in full_preprocessor.transformers_:
                if name == 'num':
                    spec["numeric_cols"] = list(cols)
                    # pipeline steps: imputer -> scaler
                    try:
                        imputer = pipe.named_steps.get('imputer')
                        scaler = pipe.named_steps.get('scaler')
                    except Exception:
                        imputer = None
                        scaler = None
                    stats = {}
                    mean_map = {}
                    scale_map = {}
                    if imputer is not None and hasattr(imputer, 'statistics_'):
                        for c, v in zip(cols, getattr(imputer, 'statistics_', [])):
                            try:
                                stats[c] = float(v) if v is not None and np.isfinite(v) else None
                            except Exception:
                                stats[c] = None
                    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        for c, m, s in zip(cols, getattr(scaler, 'mean_', []), getattr(scaler, 'scale_', [])):
                            try:
                                mean_map[c] = float(m)
                                scale_map[c] = float(s) if s not in (0, None) and np.isfinite(s) else 1.0
                            except Exception:
                                mean_map[c] = 0.0
                                scale_map[c] = 1.0
                    spec["numeric"]["imputer_statistics"] = stats
                    spec["numeric"]["scaler_mean"] = mean_map
                    spec["numeric"]["scaler_scale"] = scale_map
                elif name == 'cat':
                    spec["categorical_cols"] = list(cols)
                    try:
                        imputer = pipe.named_steps.get('imputer')
                        onehot = pipe.named_steps.get('onehot')
                    except Exception:
                        imputer = None
                        onehot = None
                    fill_map = {}
                    if imputer is not None and hasattr(imputer, 'statistics_'):
                        for c, v in zip(cols, getattr(imputer, 'statistics_', [])):
                            try:
                                fill_map[c] = None if v is None else str(v)
                            except Exception:
                                fill_map[c] = None
                    cats_map = {}
                    if onehot is not None and hasattr(onehot, 'categories_'):
                        for c, cats in zip(cols, getattr(onehot, 'categories_', [])):
                            cats_map[c] = [None if v is None else str(v) for v in list(cats)]
                    spec["categorical"]["imputer_fill"] = fill_map
                    spec["categorical"]["onehot_categories"] = cats_map
            spec_dst = os.path.join(pkg_resources_dir, "clinical_preproc_spec.json")
            with open(spec_dst, 'w') as f:
                json.dump(spec, f, indent=2)
            print(f"Saved clinical preprocessor portable spec to: {spec_dst}")
        except Exception as _e:
            print(f"Warning: failed to export portable preprocessor spec: {_e}")
        print(f"Saved final clinical MLP trained on ALL data to: {dst}")
        print(f"Saved fitted clinical preprocessor to: {preproc_dst}")
    except Exception as e:
        print(f"Warning: failed to train/save final clinical MLP on all data: {e}")

    # MLP-only: no additional base model exports

    print("\nTraining finished.")
    wandb.finish() 