from pathlib import Path
import re

import torch
from torch import nn
import soundfile as sf
from safetensors.torch import load_file
from transformers import AutoModel, AutoProcessor
from huggingface_hub import hf_hub_download

MODEL_ID = "efficient-speech/lite-whisper-large-v3-turbo-acc"
MODEL_REVISION = "ef2c0dd768cc9832a8a5a3397ab7218c838fea66"
PROCESSOR_ID = "openai/whisper-large-v3"
ADAPTER_REPO = "banhchungtuongot/ffasr-pilot-x-adapter-weights"
ADAPTER_FILE = "adapter.safetensors"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_PATTERNS = [
    r"model\.encoder\.layers\..*\.self_attn\.(q_proj|v_proj|out_proj)$",
    r"model\.decoder\.layers\..*\.encoder_attn\.(q_proj|v_proj|out_proj)$",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class LoRALinear(nn.Module):
    """Matches the training-time LoRA module exactly (rank/alpha only; inference dropout=0)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / max(self.rank, 1)
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=torch.float32))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(torch.float32), self.lora_A), self.lora_B
        ).to(out.dtype)
        return out + delta * self.scaling


def _parent_and_attr(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _attach_lora(root: nn.Module, patterns: list[str], rank: int, alpha: float) -> list[str]:
    regexes = [re.compile(p) for p in patterns]
    attached = []
    for name, module in list(root.named_modules()):
        if isinstance(module, nn.Linear) and any(r.search(name) for r in regexes):
            parent, attr = _parent_and_attr(root, name)
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
            attached.append(name)
    return attached


processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
model = AutoModel.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, torch_dtype=DTYPE,
)
model = model.to(device=DEVICE, dtype=DTYPE)
for _p in model.parameters():
    _p.requires_grad = False
_attached = _attach_lora(model, LORA_TARGET_PATTERNS, LORA_RANK, LORA_ALPHA)
if not _attached:
    raise RuntimeError("LoRA target regex matched no modules; base model layout changed?")

_adapter_path = hf_hub_download(repo_id=ADAPTER_REPO, filename=ADAPTER_FILE)
_state_dict = load_file(_adapter_path)
_missing, _unexpected = model.load_state_dict(_state_dict, strict=False)
_bad_missing = [m for m in _missing if ".lora_A" in m or ".lora_B" in m]
if _bad_missing or _unexpected:
    raise RuntimeError(f"LoRA weight load mismatch: missing={_bad_missing[:5]} unexpected={_unexpected[:5]}")
model.eval()

# Exposed for the leaderboard's parameter-count reporting.
NUM_PARAMS = sum(p.numel() for p in model.parameters())


def evaluate(file: Path) -> str:
    audio, sr = sf.read(str(file), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device=DEVICE, dtype=DTYPE)
    gen_kwargs = {"max_new_tokens": 128, "num_beams": 1, "no_repeat_ngram_size": 3}
    with torch.no_grad():
        try:
            predicted_ids = model.generate(
                input_features, language="en", task="transcribe", **gen_kwargs,
            )
        except (TypeError, ValueError):
            predicted_ids = model.generate(input_features, **gen_kwargs)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()
