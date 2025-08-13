"""
Clinical-only inference script.

Loads clinical JSON input and model artifacts from resources/ only. Supports:
- PyTorch MLP checkpoints (*.pt) saved as state_dict
- Joblib models (*.joblib), e.g., RandomSurvivalForest

No RNA-seq or histopathology inputs are loaded.
"""

from pathlib import Path
import os
import json
import math
import hashlib
import numpy
import torch

try:
    import joblib  # For scikit-survival models if present
except Exception:
    joblib = None


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


class ClinicalMLP(torch.nn.Module):
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
    # # Read the input - use thumbnail loading for tissue mask to avoid memory issues with large WSI tissue masks
    # input_tissue_mask = load_image_file_as_thumbnail(
    #     location=INPUT_PATH / "images/tissue-mask",
    #     max_size=1024,
    # )
    # # Use thumbnail loading for large WSI to avoid memory issues
    # input_bladder_cancer_tissue_biopsy_whole_slide_image = load_image_file_as_thumbnail(
    #     location=INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi",
    #     max_size=1024,
    # )
    # input_bulk_rna_seq_bladder_cancer = load_json_file(
    #     location=INPUT_PATH / "bulk-rna-seq-bladder-cancer.json",
    # )
    # input_chimera_clinical_data_of_bladder_cancer_recurrence = load_json_file(
    #     location=INPUT_PATH
    #     / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json",
    clinical_json = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
    )

    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_type, weights_path = _find_best_artifact(RESOURCE_PATH)
    if weights_path is None:
        raise FileNotFoundError(f"No model artifact found in {RESOURCE_PATH}")
    print(f"Selected artifact: type={model_type}, path={weights_path}")

    # Vectorize clinical JSON to expected input shape
    if model_type == "pt":
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        clinical_dim, hidden_dim = _infer_clinical_dims_from_state(state)
        print(f"Inferred MLP dims -> clinical_dim={clinical_dim}, hidden_dim={hidden_dim}")

        model = ClinicalMLP(input_dim=clinical_dim, hidden_dim=hidden_dim, dropout=0.4)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        clinical_vec = _vectorize_clinical(clinical_json, clinical_dim)
        with torch.no_grad():
            c_tensor = torch.from_numpy(clinical_vec).unsqueeze(0).float().to(device)
            risk_score = float(model(c_tensor).squeeze().item())

    elif model_type == "joblib":
        if joblib is None:
            raise RuntimeError("joblib is required to load the saved model but is not available")
        model = joblib.load(weights_path)
        # Try to infer required input dimension; fallback to 512
        clinical_dim = getattr(model, "n_features_", None) or 512
        clinical_vec = _vectorize_clinical(clinical_json, int(clinical_dim))
        risk_score = float(model.predict(clinical_vec.reshape(1, -1))[0])

    else:
        raise RuntimeError(f"Unsupported model artifact type: {model_type}")

    prob = 1.0 / (1.0 + math.exp(-risk_score))
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


def _infer_clinical_dims_from_state(state_dict: dict) -> tuple[int, int]:
    w0 = state_dict["network.0.weight"]  # [hidden_dim, input_dim]
    hidden_dim = int(w0.shape[0])
    clinical_dim = int(w0.shape[1])
    w3 = state_dict.get("network.3.weight")
    if w3 is not None:
        assert int(w3.shape[1]) == hidden_dim, "Mismatch in hidden layer size"
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
            idx = _stable_index(f"{k}={v}", dim)
            vec[idx] += 1.0
    return vec


def _find_best_artifact(resources_dir: Path) -> tuple[str, Path | None]:
    """Select a best-available artifact under resources/.

    Priority:
    - best_model_*.json -> read artifact name inside, prefer .pt/.joblib listed
    - clinical_mlp.pt
    - most recent best_model_*.pt or clinical_mlp_fold*.pt
    - most recent *.joblib
    Returns: (type, path) where type in {"pt", "joblib"}
    """
    # 1) Metadata JSON
    meta_files = sorted(resources_dir.glob("best_model_*.json"))
    for meta in meta_files:
        try:
            data = json.loads(meta.read_text())
            artifact = data.get("artifact")
            if artifact:
                p = resources_dir / artifact
                if p.exists():
                    if p.suffix == ".pt":
                        return "pt", p
                    if p.suffix == ".joblib":
                        return "joblib", p
        except Exception:
            pass

    # 2) Preferred single-file MLP
    preferred_pt = resources_dir / "clinical_mlp.pt"
    if preferred_pt.exists():
        return "pt", preferred_pt

    # 3) Any PT checkpoints
    pt_candidates = sorted(
        [p for p in resources_dir.glob("*.pt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pt_candidates:
        return "pt", pt_candidates[0]

    # 4) Any joblib models
    joblib_candidates = sorted(
        [p for p in resources_dir.glob("*.joblib") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if joblib_candidates:
        return "joblib", joblib_candidates[0]

    return "", None


if __name__ == "__main__":
    raise SystemExit(run())