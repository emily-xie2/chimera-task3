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
from glob import glob
import hashlib
import math
import pyvips
import numpy
import torch
import joblib


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
    w0 = state_dict["network.0.weight"]  # [hidden_dim, input_dim]
    w3 = state_dict.get("network.3.weight")  # [hidden_dim//2, hidden_dim]
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

    # Load clinical-only weights and reconstruct model
    weights_path = RESOURCE_PATH / "clinical_mlp.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

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

    # Try to load and apply CoxStack ensemble (if available)
    # CoxStack bundle must contain: {"aligned_keys": [...], "scaler": StandardScaler, "meta": CoxPHSurvivalAnalysis}
    ensemble_score = None
    try:
        coxstack_path = RESOURCE_PATH / "coxstack.joblib"
        if coxstack_path.exists():
            bundle = joblib.load(coxstack_path)
            aligned_keys = bundle.get("aligned_keys", [])
            scaler = bundle.get("scaler")
            meta = bundle.get("meta")
            if isinstance(aligned_keys, list) and scaler is not None and meta is not None and len(aligned_keys) > 0:
                # Build feature vector in the required key order. We only have MLP at inference.
                base_pred_map = {"mlp": mlp_risk_score}
                features = [float(base_pred_map.get(k, 0.0)) for k in aligned_keys]
                X = numpy.asarray(features, dtype=numpy.float32).reshape(1, -1)
                Xs = scaler.transform(X)
                # meta.predict expects same shape; returns linear predictor
                ensemble_score = float(meta.predict(Xs)[0])
    except Exception as e:
        print(f"Warning: CoxStack inference failed, falling back to MLP only ({e})")

    final_score = ensemble_score if ensemble_score is not None else mlp_risk_score

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
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    file_path = input_files[0]
    print(f"Loading pathology image as thumbnail using PyVips (fast path): {file_path}")
    try:
        # Fast path: decode at thumbnail size using pyramid levels when available
        thumb = pyvips.Image.thumbnail(file_path, max_size, height=max_size)
        return thumb
    except Exception as e:
        print(f"PyVips thumbnail fast path failed ({e}), falling back to sequential read")
        image = pyvips.Image.new_from_file(file_path, access="sequential")
        scale_factor = min(max_size / image.width, max_size / image.height)
        if scale_factor < 1.0:
            print(
                f"Downsampling image by factor {scale_factor:.3f} (from {image.width}x{image.height} to {int(image.width*scale_factor)}x{int(image.height*scale_factor)})"
            )
            image = image.resize(scale_factor)
        else:
            print(
                f"Image size {image.width}x{image.height} is within max_size={max_size}, no downsampling needed"
            )
        return image


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