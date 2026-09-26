#!/usr/bin/env python3
"""
dump_reference_nemotron3_diar_transformers.py - Nemotron-3 Diarization
(Streaming Sortformer v3) reference tensors from Hugging Face Transformers
(`Nemotron3DiarizationForAudioFrameClassification`).

Like Sortformer v2 this is a frame-level end-to-end neural diarizer
(`encoder-diarizer`), not a transcription model: the behavioral artifact is
a T x 8 speaker-activity probability matrix at the 10 ms mel frame rate.

Pipeline (one chunk, full context):
    mel (128 x T_mel, preemph 0.97, log, no normalization)
    -> embedder        stack 8 mel frames, Linear 1024 -> 512 (no bias)
    -> input_layer_norm
    -> 31 pre-LN Transformer layers (RoPE, GELU MLP 2048)
    -> layer_norm
    -> proj            Linear 512 -> 192
    -> upsampler       Conv1d 192 -> 192*8 (k=3, pad=1), sub-pixel reshape
    -> classifier      relu -> Linear 192 -> relu -> Linear 8
    -> sigmoid         probs [8*T_enc, 8]

Two reference paths are dumped:
  * `encoder`  one full-context forward with no speaker cache (the
               non-stateful graph the C++ port must match first).
  * `diarize`  the chunked AOSC/FIFO forward at a latency preset, the
               runtime target, trimmed to the mel frame count.

    uv run --project scripts/envs/nemotron3_diar \
      scripts/dump_reference_nemotron3_diar_transformers.py encoder \
      --model nvidia/Nemotron-3-Diarization --audio samples/sortformer-2spk-mix.wav \
      --out build/validate/nemotron3_diar/nemotron-3-diarization/sortformer-2spk-mix/encoder/ref
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ref_dump  # noqa: E402

write_tensor = ref_dump.write_tensor
write_transcript = ref_dump.write_transcript

# Latency presets in 80 ms encoder frames: chunk / right context / fifo /
# update period. The speaker cache is 264 frames throughout. `offline` is the
# checkpoint's offline-mode config; the three low-latency rows are NVIDIA's
# streaming presets (processor_config.json streaming_modes plus the
# streaming_config FIFO sizes). `small` is a diagnostic that forces several
# chunks and speaker-cache compression on a short clip. Keep in sync with
# k_presets in src/arch/nemotron3_diar/stream.cpp.
PRESETS = {
    "offline": dict(chunk=340, rc=40, fifo=40, update=300, cache=264),
    "low_latency": dict(chunk=9, rc=4, fifo=264, update=222, cache=264),
    "very_low_latency": dict(chunk=6, rc=2, fifo=264, update=222, cache=264),
    "ultra_low_latency": dict(chunk=3, rc=1, fifo=264, update=222, cache=264),
    "small": dict(chunk=12, rc=2, fifo=8, update=6, cache=24),
}


def _load(model: str):
    from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification

    processor = AutoProcessor.from_pretrained(model)
    m = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
        model, dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()
    return processor, m


def _read_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {sr} Hz")
    return audio.astype(np.float32)


def _features(processor, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Mel features [1, T_mel, 128] and the valid-frame mask [1, T_mel]. The
    trailing center-padding frame is zeroed and masked, as in the official
    processor -> model(**inputs) pipeline."""
    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_features"].to(torch.float32), inputs["attention_mask"]


def _src(model: str, hook: str, **extra: Any) -> dict[str, Any]:
    return {"framework": "transformers", "model": model, "hook": hook, **extra}


def _np(t: torch.Tensor) -> np.ndarray:
    a = t.detach().to(torch.float32).cpu().numpy()
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a.astype(np.float32))


def cmd_encoder(args: argparse.Namespace) -> int:
    processor, m = _load(args.model)
    audio = _read_audio(args.audio)
    out_dir = Path(args.out)
    feats, mask = _features(processor, audio)
    n_mel = feats.shape[1]

    # Native NeMo layout [n_mels, T_mel], matching the C++ mel dump.
    write_tensor("enc.mel.in", np.ascontiguousarray(_np(feats).T), "frontend",
                 _src(args.model, "feature_extractor.input_features"), out_dir=out_dir)

    tower = m.model.audio_tower
    captured: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_mod, _inp, out):
            captured[name] = out[0] if isinstance(out, (tuple, list)) else out
        return hook

    handles = [
        tower.embedder.register_forward_hook(grab("embed")),
        tower.layers[0].register_forward_hook(grab("layer0")),
        tower.layer_norm.register_forward_hook(grab("audio")),
        m.model.proj.register_forward_hook(grab("proj")),
        m.model.upsampler.register_forward_hook(grab("upsampled")),
    ]
    try:
        with torch.no_grad():
            hidden = m.model(input_features=feats, attention_mask=mask).last_hidden_state
            logits = m.classifier(hidden)
    finally:
        for h in handles:
            h.remove()

    src = lambda hook: _src(args.model, hook)  # noqa: E731
    write_tensor("enc.embed.out", _np(captured["embed"]), "encoder", src("audio_tower.embedder"), out_dir=out_dir)
    write_tensor("enc.layer0.out", _np(captured["layer0"]), "encoder", src("audio_tower.layers.0"), out_dir=out_dir)
    write_tensor("enc.audio.out", _np(captured["audio"]), "encoder", src("audio_tower.layer_norm"), out_dir=out_dir)
    write_tensor("enc.proj.out", _np(captured["proj"]), "encoder", src("model.proj"), out_dir=out_dir)
    write_tensor("enc.upsample.out", _np(captured["upsampled"]), "encoder", src("model.upsampler"), out_dir=out_dir)
    probs = torch.sigmoid(logits)[:, :n_mel]
    write_tensor("diar.preds_offline", _np(probs), "encoder",
                 src("classifier.sigmoid(full-context, no cache)"), out_dir=out_dir)
    print(f"wrote encoder-stage tensors ({n_mel} mel frames) to {out_dir}")
    return 0


def run_preset(m, feats: torch.Tensor, mask: torch.Tensor, preset: str) -> torch.Tensor:
    """Chunked AOSC/FIFO forward at a preset, via the offline-mode forward
    with the chunk geometry overridden. Returns sigmoid probs [T_mel, 8]."""
    p = PRESETS[preset]
    cfg = m.config
    saved = (cfg.chunk_length, cfg.chunk_right_context, cfg.fifo_length, cfg.speaker_cache_update_period,
             cfg.streaming_config.speaker_cache_length)
    cfg.chunk_length, cfg.chunk_right_context = p["chunk"], p["rc"]
    cfg.fifo_length, cfg.speaker_cache_update_period = p["fifo"], p["update"]
    cfg.streaming_config.speaker_cache_length = p["cache"]
    try:
        with torch.no_grad():
            logits = m(input_features=feats, attention_mask=mask).logits
    finally:
        (cfg.chunk_length, cfg.chunk_right_context, cfg.fifo_length, cfg.speaker_cache_update_period,
         cfg.streaming_config.speaker_cache_length) = saved
    return torch.sigmoid(logits)[0]


def cmd_diarize(args: argparse.Namespace) -> int:
    processor, m = _load(args.model)
    audio = _read_audio(args.audio)
    out_dir = Path(args.out)
    feats, mask = _features(processor, audio)
    probs = _np(run_preset(m, feats, mask, args.preset))
    write_tensor("diar.probs", probs, "diarize",
                 _src(args.model, "forward.sigmoid(chunked AOSC/FIFO)", preset=args.preset,
                      geometry=PRESETS[args.preset]),
                 out_dir=out_dir)

    lines = []
    active = probs > 0.5
    for spk in range(probs.shape[1]):
        start = None
        for t in range(probs.shape[0] + 1):
            on = t < probs.shape[0] and active[t, spk]
            if on and start is None:
                start = t
            elif not on and start is not None:
                lines.append(f"{start * 0.01:.2f} {t * 0.01:.2f} speaker_{spk}")
                start = None
    write_transcript(out_dir, "\n".join(lines), source=_src(args.model, "threshold(0.5) segments"))
    print(f"wrote diarize probs [{probs.shape[0]}x{probs.shape[1]}] ({args.preset}) + {len(lines)} segments")
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint directory")
    p.add_argument("--audio", required=True, help="Path to a 16 kHz mono WAV")
    p.add_argument("--out", required=True, help="Output ref/ directory")
    p.add_argument("--torch-threads", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--language", default=None, help=argparse.SUPPRESS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ep = sub.add_parser("encoder", help="Dump mel + per-stage activations (full-context forward)")
    add_common_args(ep)
    ep.set_defaults(func=cmd_encoder)
    dp = sub.add_parser("diarize", help="Dump chunked AOSC/FIFO probs at a latency preset")
    add_common_args(dp)
    dp.add_argument("--preset", default="offline", choices=list(PRESETS))
    dp.set_defaults(func=cmd_diarize)
    args = p.parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
