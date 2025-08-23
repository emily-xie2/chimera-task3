"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import os
import json
import hashlib
import math
import warnings
import numpy
import torch

# Suppress sklearn version mismatch warnings from unpickling
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


# -----------------------------
# Minimal model architecture
# -----------------------------
class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionMLP(torch.nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        rna_dim: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.clinical_enc = MLPEncoder(clinical_dim, hidden_dim, embed_dim, dropout)
        self.rna_enc = MLPEncoder(rna_dim, hidden_dim, embed_dim, dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, clinical_x: torch.Tensor, rna_x: torch.Tensor) -> torch.Tensor:
        c = self.clinical_enc(clinical_x)
        r = self.rna_enc(rna_x)
        fused = torch.cat([c, r], dim=1)
        return self.head(fused)


class ClinicalMLP(torch.nn.Module):
    """Clinical-only MLP matching training architecture.

    Architecture: Linear(input_dim -> hidden_dim) -> ReLU -> Dropout ->
                  Linear(hidden_dim -> hidden_dim//2) -> ReLU -> Dropout ->
                  Linear(hidden_dim//2 -> 1)
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _remap_clinical_state_if_needed(state_dict: dict) -> dict:
    """Remap exported clinical MLP keys (fc1/fc2/out) to inference model keys (network.*).

    Training export may save keys as fc1/fc2/out. This converts them to the
    ClinicalMLP's expected sequential keys.
    """
    if isinstance(state_dict, dict) and "fc1.weight" in state_dict:
        key_map = {
            "fc1.weight": "network.0.weight",
            "fc1.bias": "network.0.bias",
            "fc2.weight": "network.3.weight",
            "fc2.bias": "network.3.bias",
            "out.weight": "network.6.weight",
            "out.bias": "network.6.bias",
        }
        remapped = {}
        for src_key, dst_key in key_map.items():
            if src_key in state_dict:
                remapped[dst_key] = state_dict[src_key]
        return remapped
    return state_dict


def _infer_model_dims_from_state(state_dict: dict) -> tuple[int, int, int, int]:
    """Infer (clinical_dim, rna_dim, embed_dim, hidden_dim) from state shapes."""
    # clinical encoder first and second linear layers
    w_c0 = state_dict["clinical_enc.net.0.weight"]  # [hidden_dim, clinical_dim]
    w_c3 = state_dict["clinical_enc.net.3.weight"]  # [embed_dim, hidden_dim]
    # rna encoder first linear
    w_r0 = state_dict["rna_enc.net.0.weight"]  # [hidden_dim, rna_dim]
    # head first linear
    w_h0 = state_dict["head.0.weight"]  # [hidden_dim, 2*embed_dim]

    hidden_dim = w_c0.shape[0]
    clinical_dim = w_c0.shape[1]
    embed_dim = w_c3.shape[0]
    rna_dim = w_r0.shape[1]
    assert w_h0.shape[1] == 2 * embed_dim, "Head expects 2*embed_dim inputs"
    return clinical_dim, rna_dim, embed_dim, hidden_dim


def _infer_clinical_dims_from_state(state_dict: dict) -> tuple[int, int]:
    """Infer (clinical_dim, hidden_dim) from clinical-only state shapes."""
    # Support both naming schemes: network.* and fc*/out.*
    if "network.0.weight" in state_dict:
        w0 = state_dict["network.0.weight"]  # [hidden_dim, input_dim]
        w3 = state_dict.get("network.3.weight")  # [hidden_dim//2, hidden_dim]
    elif "fc1.weight" in state_dict:
        w0 = state_dict["fc1.weight"]  # [hidden_dim, input_dim]
        w3 = state_dict.get("fc2.weight")  # [hidden_dim//2, hidden_dim]
    else:
        raise KeyError("Unsupported clinical MLP state dict format: missing 'network.0.weight' or 'fc1.weight'")
    hidden_dim = w0.shape[0]
    clinical_dim = w0.shape[1]
    if w3 is not None:
        assert w3.shape[1] == hidden_dim, "Mismatch in hidden layer size"
    return clinical_dim, hidden_dim


def _stable_index(key: str, dim: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def _vectorize_clinical(clinical_json: dict, dim: int) -> numpy.ndarray:
    vec = numpy.zeros(dim, dtype=numpy.float32)
    if not isinstance(clinical_json, dict):
        return vec
    for k, v in clinical_json.items():
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            idx = _stable_index(k, dim)
            vec[idx] += float(v)
        else:
            # hash categorical as key=value
            idx = _stable_index(f"{k}={v}", dim)
            vec[idx] += 1.0
    return vec


def _flatten_numeric_items(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_numeric_items(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten_numeric_items(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], obj


def _vectorize_rna(rna_json: dict, dim: int) -> numpy.ndarray:
    vec = numpy.zeros(dim, dtype=numpy.float32)
    if not isinstance(rna_json, dict):
        return vec
    # Many datasets are {gene: expression}
    # But handle nested structures robustly
    for k, v in _flatten_numeric_items(rna_json):
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            idx = _stable_index(k, dim)
            vec[idx] += float(v)
    return vec


def _apply_portable_preprocessor(clinical_json: dict, spec: dict) -> numpy.ndarray:
    """Apply portable preprocessor spec to a single clinical JSON row.

    Returns a 1D numpy array matching the training feature order:
    [standardized numeric..., onehot categorical...]
    """
    num_cols = spec.get('numeric_cols', [])
    cat_cols = spec.get('categorical_cols', [])
    num_stats = spec.get('numeric', {})
    cat_stats = spec.get('categorical', {})
    imputer_stats = (num_stats or {}).get('imputer_statistics', {})
    scaler_mean = (num_stats or {}).get('scaler_mean', {})
    scaler_scale = (num_stats or {}).get('scaler_scale', {})
    onehot_cats = (cat_stats or {}).get('onehot_categories', {})

    # Numeric block
    num_array = []
    for c in num_cols:
        raw = clinical_json.get(c, None)
        if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
            raw = imputer_stats.get(c, 0.0)
        try:
            x = float(raw)
        except Exception:
            x = 0.0
        m = float(scaler_mean.get(c, 0.0))
        s = float(scaler_scale.get(c, 1.0))
        if s == 0.0 or not math.isfinite(s):
            s = 1.0
        num_array.append((x - m) / s)
    num_array = numpy.asarray(num_array, dtype=numpy.float32)

    # Categorical one-hot block
    oh_vectors = []
    for c in cat_cols:
        cats = onehot_cats.get(c, [])
        v = clinical_json.get(c, None)
        v_str = None if v is None else str(v)
        row = [1.0 if (v_str == (None if cat is None else str(cat))) else 0.0 for cat in cats]
        oh_vectors.append(numpy.asarray(row, dtype=numpy.float32))
    if oh_vectors:
        oh_array = numpy.concatenate(oh_vectors, axis=0)
    else:
        oh_array = numpy.zeros((0,), dtype=numpy.float32)

    feats = numpy.concatenate([num_array, oh_array], axis=0)
    return feats


def run():
    interface_key = get_interface_key()
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "bulk-rna-seq-bladder-cancer",
            "chimera-clinical-data-of-bladder-cancer-recurrence",
            "tissue-mask",
        ): interf0_handler,
    }[interface_key]
    return handler()


def interf0_handler():
    clinical_json = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
    )

    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load clinical-only MLP (required)
    weights_path = RESOURCE_PATH / "clinical_mlp.pt"
    if not weights_path.exists():
        raise RuntimeError(f"Required clinical MLP weights not found at {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Remap state keys if exported as fc1/fc2/out
    state = _remap_clinical_state_if_needed(state)

    clinical_dim, hidden_dim = _infer_clinical_dims_from_state(state)
    print(f"Inferred clinical dims -> clinical_dim={clinical_dim}, hidden_dim={hidden_dim}")

    model = ClinicalMLP(input_dim=clinical_dim, hidden_dim=hidden_dim, dropout=0.4)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Vectorize clinical inputs via deterministic hashing to match expected dims
    clinical_vec = _vectorize_clinical(clinical_json, clinical_dim)

    with torch.no_grad():
        c_tensor = torch.from_numpy(clinical_vec).unsqueeze(0).float().to(device)
        mlp_risk_score = model(c_tensor).squeeze().item()

    # Require MLP+RSF ensemble bundle (no fallback)
    stats = {}
    aligned_keys = []
    mlp_rsf_path = RESOURCE_PATH / "mlp_rsf_ensemble.json"
    if not mlp_rsf_path.exists():
        raise RuntimeError(f"Required MLP+RSF ensemble bundle not found at {mlp_rsf_path}")
    with open(mlp_rsf_path, "r") as f:
        bundle = json.load(f)
    aligned_keys = bundle.get("aligned_keys", [])
    stats = bundle.get("stats", {})
    if not isinstance(aligned_keys, list) or not aligned_keys:
        raise RuntimeError("MLP+RSF ensemble bundle missing 'aligned_keys'")

    # Build portable base predictions
    base_pred_map: dict[str, float] = {}
    base_pred_map['mlp'] = float(mlp_risk_score)

    # Compute CoxPH base prediction using portable JSON and portable preprocessor spec
    spec_path = RESOURCE_PATH / "clinical_preproc_spec.json"
    cox_portable_path = RESOURCE_PATH / "coxph_portable.json"
    if not spec_path.exists():
        raise RuntimeError(f"Required portable preprocessor spec not found at {spec_path}")
    if not cox_portable_path.exists():
        raise RuntimeError(f"Required portable CoxPH weights not found at {cox_portable_path}")
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    with open(cox_portable_path, 'r') as f:
        cox = json.load(f)
    x_pre = _apply_portable_preprocessor(clinical_json, spec)
    c_mean = numpy.asarray(cox.get('scaler_mean', []), dtype=numpy.float32)
    c_scale = numpy.asarray(cox.get('scaler_scale', []), dtype=numpy.float32)
    beta = numpy.asarray(cox.get('coef', []), dtype=numpy.float32)
    if x_pre.shape[0] != c_mean.shape[0] or x_pre.shape[0] != c_scale.shape[0] or x_pre.shape[0] != beta.shape[0]:
        raise RuntimeError(
            f"CoxPH portable shapes mismatch: x={x_pre.shape[0]}, mean={c_mean.shape[0]}, scale={c_scale.shape[0]}, coef={beta.shape[0]}"
        )
    # Apply CoxPH scaler and compute linear predictor
    x_scaled = (x_pre - c_mean) / numpy.where(c_scale == 0.0, 1.0, c_scale)
    coxph_score = float(numpy.dot(x_scaled, beta))
    base_pred_map['coxph'] = coxph_score

    # Optional: RSF base prediction via joblib bundle if present (still weight-only ensemble)
    try:
        rsf_bundle_path = RESOURCE_PATH / "rsf_full.joblib"
        if rsf_bundle_path.exists() and 'rsf' in aligned_keys:
            import joblib  # local import to avoid global dependency unless needed
            rsf_obj = joblib.load(rsf_bundle_path)
            rsf_model = rsf_obj.get('model') if isinstance(rsf_obj, dict) else rsf_obj
            # Reuse the same portable features used for CoxPH
            rsf_score = float(numpy.atleast_1d(rsf_model.predict(x_pre.reshape(1, -1)))[0])
            base_pred_map['rsf'] = rsf_score
    except Exception as _e:
        print(f"Warning: failed to compute RSF base prediction: {_e}")

    # Only allow supported keys and require at least 2
    supported = ['mlp', 'coxph', 'rsf']
    used_keys = [k for k in aligned_keys if k in supported]
    if len(used_keys) < 2:
        raise RuntimeError("Ensemble must include at least two supported keys, e.g., 'mlp' and 'coxph' or 'rsf'")
    # Assemble z-scored features for used keys
    feats = []
    for k in used_keys:
        if k not in stats or not isinstance(stats.get(k), (list, tuple)) or len(stats.get(k)) != 2:
            raise RuntimeError(f"TopKMean stats for '{k}' are missing or malformed")
        mu, sd = stats[k]
        sd = float(sd) if (isinstance(sd, (int, float)) and float(sd) > 1e-8) else 1.0
        feats.append((float(base_pred_map[k]) - float(mu)) / sd)
    final_score = float(sum(feats) / len(feats))
    print(f"Ensemble used keys: {used_keys}")

    # Map final_score to [0, 80] using sigmoid
    prob = 1.0 / (1.0 + math.exp(-final_score))
    output_likelihood_of_bladder_cancer_recurrence = round(80.0 * prob, 1)

    write_json_file(
        location=OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json",
        content=output_likelihood_of_bladder_cancer_recurrence,
    )
    return 0


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_thumbnail(*, location, max_size=1024):
    raise NotImplementedError("Image loading is not used in this inference pipeline")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"\tcurrent device: {current_device}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())