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
import json
import torch
import torch.nn as nn

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "bulk-rna-seq-bladder-cancer",
            "chimera-clinical-data-of-bladder-cancer-recurrence",
            "tissue-mask",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    """
    Clinical-only inference handler. Reads clinical JSON, encodes features,
    runs a small MLP (weights loaded if available), and writes likelihood [0, 100].
    """
    _show_torch_cuda_info()

    clinical = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json",
    )

    x = encode_clinical(clinical)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try loading trained weights and infer model dims if available
    load_result = _load_trained_weights()
    if load_result is not None:
        state_dict, inferred_in_dim, inferred_hidden_dim = load_result
        model = ClinicalMLP(input_dim=inferred_in_dim, hidden_dim=inferred_hidden_dim, dropout=0.4)
        model.load_state_dict(state_dict)
    else:
        model = ClinicalMLP(input_dim=x.shape[0], hidden_dim=64, dropout=0.4)

    # Adjust input feature vector to expected dimension (pad with zeros or truncate)
    expected_in = model.fc1.in_features
    if x.shape[0] < expected_in:
        pad = torch.zeros(expected_in - x.shape[0], dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    elif x.shape[0] > expected_in:
        x = x[:expected_in]

    x = x.unsqueeze(0).to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        likelihood = round(float(prob) * 100.0, 1)

    write_json_file(
        location=OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json",
        content=likelihood,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        # Match utils.models.mlp.PredictionModel_Clinical architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.out(x)
        return x


def _load_trained_weights():
    """
    Search for a saved state_dict and infer model dimensions from the first layer
    weights. Returns (state_dict, input_dim, hidden_dim) or None.
    """
    candidate_paths = [
        Path("/opt/app/resources") / "clinical_mlp.pt",
        RESOURCE_PATH / "clinical_mlp.pt",
    ]
    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            print(f"Found clinical model weights at: {p}")
            state = torch.load(p, map_location="cpu")
            # Try common key patterns to infer dimensions
            key_options = [
                "network.0.weight",   # utils.models.mlp.PredictionModel_Clinical
                "fc1.weight",         # this module naming
            ]
            for k in key_options:
                if k in state:
                    w = state[k]
                    hidden_dim, in_dim = int(w.shape[0]), int(w.shape[1])
                    return state, in_dim, hidden_dim
            # Fallback: try to parse any first Linear weight
            for k, v in state.items():
                if k.endswith(".weight") and len(v.shape) == 2:
                    hidden_dim, in_dim = int(v.shape[0]), int(v.shape[1])
                    return state, in_dim, hidden_dim
            print(f"Warning: could not infer model dims from state dict keys at {p}")
        except Exception as e:
            print(f"Warning: failed to load/inspect weights from {p}: {e}")
    print("No clinical model weights found.")
    return None


def encode_clinical(cd):
    """
    One-hot encodes categorical clinical features and appends normalized numerical features.
    """
    cat_map = {
        "sex": ["Male", "Female"],
        "smoking": ["No", "Yes"],
        "tumor": ["Primary", "Recurrence"],
        "stage": ["TaHG", "T1HG", "T2HG"],
        "substage": ["T1m", "T1e"],
        "grade": ["G2", "G3"],
        "reTUR": ["No", "Yes"],
        "LVI": ["No", "Yes"],
        "variant": ["UCC", "UCC + Variant"],
        "EORTC": ["High risk", "Highest risk"],
        "BRS": ["BRS1", "BRS2", "BRS3"],
    }

    one_hot = []
    for key, options in cat_map.items():
        vec = [0] * len(options)
        value = cd.get(key, None)
        if value in options:
            vec[options.index(value)] = 1
        else:
            vec[0] = 1
        one_hot.extend(vec)

    # Numerical features (simple normalization/scaling)
    try:
        age = float(cd.get("age", 0.0)) / 100.0
    except Exception:
        age = 0.0
    try:
        instills = float(cd.get("no_instillations", -1.0))
    except Exception:
        instills = -1.0

    numerical = [age, instills]

    return torch.tensor(numerical + one_hot, dtype=torch.float32)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())