from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


root = Path(__file__).resolve().parent.parent / "Models"
root.mkdir(exist_ok=True)

models = [
    ("mlx-community/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja"),
    ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3"),
    ("mlx-community/Hy-MT2-1.8B-8bit", "Hy-MT2-1.8B-8bit"),
]
choice = root / "translation-model.txt"
chosen = choice.read_text(encoding="utf-8").strip() if choice.is_file() else ""
if chosen and chosen not in [name for _, name in models]:
    models.append((f"mlx-community/{chosen}", chosen))

for model, name in models:
    snapshot_download(repo_id=model, local_dir=root / name, ignore_patterns=["mlx_model/*"])

(root / "silero-vad").mkdir(exist_ok=True)
vad = hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx")
(root / "silero-vad" / "model.onnx").write_bytes(Path(vad).read_bytes())
