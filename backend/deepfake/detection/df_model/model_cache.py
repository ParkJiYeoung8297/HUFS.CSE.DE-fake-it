from pathlib import Path
from threading import Lock

import torch

from .model import Model


checkpoint_path = Path(__file__).resolve().parent

_model_cache = {}
_cache_lock = Lock()


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )


def get_cached_model(
    purpose,
    selected_model='EfficientNet-b0',
    checkpoint_name='checkpoint_v35',
):
    device = get_device()
    cache_key = (purpose, selected_model, checkpoint_name, str(device))

    with _cache_lock:
        if cache_key not in _model_cache:
            print(f"Loading {purpose} model once: {selected_model}/{checkpoint_name} on {device}")
            model = Model(
                num_binary_classes=2,
                num_method_classes=7,
                model_name=selected_model
            ).to(device)
            model.load_state_dict(
                torch.load(f'{checkpoint_path}/{checkpoint_name}.pt', map_location=device)
            )
            model.eval()
            _model_cache[cache_key] = {
                "model": model,
                "device": device,
                "lock": Lock(),
            }

    cached = _model_cache[cache_key]
    return cached["model"], cached["device"], cached["lock"]


def preload_cached_models(
    selected_model='EfficientNet-b0',
    checkpoint_name='checkpoint_v35',
):
    get_cached_model("inference", selected_model, checkpoint_name)
    get_cached_model("gradcam", selected_model, checkpoint_name)
