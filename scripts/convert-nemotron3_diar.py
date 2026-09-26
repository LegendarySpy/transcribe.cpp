#!/usr/bin/env python3
"""
convert-nemotron3_diar.py - convert NVIDIA Nemotron-3 Diarization (Streaming
Sortformer v3, HF Transformers `Nemotron3DiarizationForAudioFrameClassification`
safetensors) into a reference-dtype GGUF.

`encoder-diarizer`: no tokenizer, no decoder, no text. Tensor sources
(417 tensors in model.safetensors):

    model.audio_tower.embedder.projection      -> enc.embed.proj           (Linear 8*128 -> 512, no bias)
    model.audio_tower.input_layer_norm         -> enc.norm_in
    model.audio_tower.layers.{i}.*             -> enc.blocks.{i}.*         (31 pre-LN RoPE Transformer layers)
    model.audio_tower.layer_norm               -> enc.norm_out
    model.proj                                 -> diar.proj                (Linear 512 -> 192)
    model.upsampler.conv (Conv1d k=3)          -> diar.upsample.tap{0,1,2} (one 192 -> 1536 Linear per tap)
    classifier.dense / classifier.out_proj     -> diar.head.dense / diar.head.out
    silence_embeds                             -> diar.silence_emb         (learned AOSC silence slot, F32)

Reference dtype is F32 (the checkpoint is fp32).

    uv run --project scripts/envs/nemotron3_diar \
      scripts/convert-nemotron3_diar.py nvidia/Nemotron-3-Diarization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    gguf_writer,
    reference_dtype_for,
)

ARCH = "nemotron3_diar"
VARIANT = "nemotron-3-diarization"
REFERENCE_TYPE = GGMLQuantizationType.F32
REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = 0  # LlamaFileType ALL_F32

TOP_TABLE = [
    ("model.audio_tower.embedder.projection.weight", "enc.embed.proj.weight"),
    ("model.audio_tower.input_layer_norm.weight", "enc.norm_in.weight"),
    ("model.audio_tower.input_layer_norm.bias", "enc.norm_in.bias"),
    ("model.audio_tower.layer_norm.weight", "enc.norm_out.weight"),
    ("model.audio_tower.layer_norm.bias", "enc.norm_out.bias"),
    ("model.proj.weight", "diar.proj.weight"),
    ("model.proj.bias", "diar.proj.bias"),
    ("model.upsampler.conv.bias", "diar.upsample.bias"),
    ("classifier.dense.weight", "diar.head.dense.weight"),
    ("classifier.dense.bias", "diar.head.dense.bias"),
    ("classifier.out_proj.weight", "diar.head.out.weight"),
    ("classifier.out_proj.bias", "diar.head.out.bias"),
    ("silence_embeds", "diar.silence_emb"),
]

BLOCK_TABLE = [
    ("layer_norm1.weight", "norm_attn.weight"),
    ("layer_norm1.bias", "norm_attn.bias"),
    ("self_attn.q_proj.weight", "attn.q.weight"),
    ("self_attn.k_proj.weight", "attn.k.weight"),
    ("self_attn.v_proj.weight", "attn.v.weight"),
    ("self_attn.o_proj.weight", "attn.out.weight"),
    ("self_attn.o_proj.bias", "attn.out.bias"),
    ("layer_norm2.weight", "norm_ff.weight"),
    ("layer_norm2.bias", "norm_ff.bias"),
    ("mlp.fc1.weight", "ff.in.weight"),
    ("mlp.fc1.bias", "ff.in.bias"),
    ("mlp.fc2.weight", "ff.out.weight"),
    ("mlp.fc2.bias", "ff.out.bias"),
]

UPSAMPLE_WEIGHT = "model.upsampler.conv.weight"


def _load(model_spec: str) -> tuple[dict, dict, dict, str | None]:
    from huggingface_hub import hf_hub_download, model_info
    from safetensors.numpy import load_file

    local = Path(model_spec)
    if local.is_dir():
        weights = local / "model.safetensors"
        config = json.loads((local / "config.json").read_text())
        processor = json.loads((local / "processor_config.json").read_text())
        revision = None
    else:
        revision = model_info(model_spec).sha
        weights = Path(hf_hub_download(model_spec, "model.safetensors", revision=revision))
        config = json.loads(Path(hf_hub_download(model_spec, "config.json", revision=revision)).read_text())
        processor = json.loads(
            Path(hf_hub_download(model_spec, "processor_config.json", revision=revision)).read_text()
        )
    return load_file(str(weights)), config, processor, revision


def _add(writer, name: str, arr: np.ndarray) -> None:
    if arr.dtype != np.float32:
        raise ValueError(f"{name}: expected fp32, got {arr.dtype}")
    ggml_type = reference_dtype_for(name, REFERENCE_TYPE)
    data, out_type = encode_for_gguf(np.ascontiguousarray(arr), ggml_type)
    writer.add_tensor(name, data, raw_dtype=out_type)


def convert(model_spec: str, out_path: Path, repo_id: str | None) -> None:
    sd, cfg, proc, revision = _load(model_spec)
    audio = cfg["audio_config"]
    head = cfg["head_config"]
    stream = cfg["streaming_config"]
    fe = proc["feature_extractor"]

    n_layers = int(audio["num_hidden_layers"])
    d_model = int(audio["hidden_size"])
    n_heads = int(audio["num_attention_heads"])
    if int(audio["num_key_value_heads"]) != n_heads:
        raise ValueError("grouped-query attention is not expected in this encoder")
    if float(audio["rope_parameters"].get("partial_rotary_factor", 1.0)) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    if audio["hidden_act"] != "gelu":
        raise ValueError(f"unexpected activation {audio['hidden_act']}")
    sub = int(audio["subsampling_factor"])
    hop = int(fe["hop_length"])
    sr = int(fe["sampling_rate"])
    n_spk = int(head["num_speakers"])
    print(f"layers={n_layers} d_model={d_model} heads={n_heads} speakers={n_spk} subsampling={sub}")

    writer = gguf_writer(str(out_path), ARCH)
    add_general_identity(
        writer,
        name="Nemotron-3 Diarization",
        basename="nemotron-3-diarization",
        size_label="100M",
        version="v3",
        file_type=REFERENCE_FILE_TYPE,
        languages=["en"],
        author="NVIDIA",
        organization="nvidia",
        license="other",
        license_name="openmdw-1.1",
        license_link="https://huggingface.co/nvidia/Nemotron-3-Diarization",
        repo_url=f"https://huggingface.co/{repo_id}" if repo_id else None,
        description="End-to-end streaming speaker diarizer (Streaming Sortformer v3): RoPE Transformer encoder, "
        "up to 8 speakers, speaker activity every 10 ms.",
    )
    writer.add_string("stt.variant", out_path.parent.name)
    if revision:
        writer.add_string("stt.source.hf_revision", revision)

    writer.add_uint32("stt.frontend.sample_rate", sr)
    writer.add_uint32("stt.frontend.num_mels", int(fe["feature_size"]))
    writer.add_uint32("stt.frontend.n_fft", int(fe["n_fft"]))
    writer.add_uint32("stt.frontend.hop_length", hop)
    writer.add_uint32("stt.frontend.win_length", int(fe["win_length"]))
    writer.add_string("stt.frontend.window", "hann")
    writer.add_string("stt.frontend.normalize", "none")
    writer.add_float32("stt.frontend.pre_emphasis", float(fe["preemphasis"]))
    writer.add_float32("stt.frontend.dither", 0.0)

    writer.add_bool("stt.capability.streaming", True)
    writer.add_bool("stt.capability.speaker_diarization", True)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.timestamps", False)

    p = "stt.nemotron3_diar"
    writer.add_uint32(f"{p}.max_speakers", n_spk)
    writer.add_uint32(f"{p}.frame_hop", hop)
    writer.add_uint32(f"{p}.subsampling_factor", sub)
    writer.add_uint32(f"{p}.encoder.n_layers", n_layers)
    writer.add_uint32(f"{p}.encoder.d_model", d_model)
    writer.add_uint32(f"{p}.encoder.n_heads", n_heads)
    writer.add_uint32(f"{p}.encoder.d_ff", int(audio["intermediate_size"]))
    writer.add_float32(f"{p}.encoder.rope_theta", float(audio["rope_parameters"]["rope_theta"]))
    writer.add_uint32(f"{p}.head.d_model", int(head["hidden_size"]))

    writer.add_uint32(f"{p}.stream.offline.chunk_len", int(cfg["chunk_length"]))
    writer.add_uint32(f"{p}.stream.offline.right_context", int(cfg["chunk_right_context"]))
    writer.add_uint32(f"{p}.stream.offline.fifo_len", int(cfg["fifo_length"]))
    writer.add_uint32(f"{p}.stream.offline.spkcache_update_period", int(cfg["speaker_cache_update_period"]))
    writer.add_uint32(f"{p}.stream.fifo_len", int(stream["fifo_length"]))
    writer.add_uint32(f"{p}.stream.spkcache_update_period", int(stream["speaker_cache_update_period"]))
    writer.add_uint32(f"{p}.stream.spkcache_len", int(stream["speaker_cache_length"]))
    writer.add_uint32(f"{p}.stream.sil_frames_per_spk", int(stream["speaker_cache_silence_frames_per_speaker"]))
    writer.add_float32(f"{p}.stream.pred_score_threshold", float(stream["prediction_score_threshold"]))
    writer.add_float32(f"{p}.stream.latest_frames_score_boost", float(stream["latest_frames_score_boost"]))
    writer.add_float32(f"{p}.stream.strong_boost_rate", float(stream["strong_boost_rate"]))
    writer.add_float32(f"{p}.stream.weak_boost_rate", float(stream["weak_boost_rate"]))
    writer.add_float32(f"{p}.stream.min_pos_scores_rate", float(stream["min_positive_scores_rate"]))

    used: set[str] = set()

    def emit(src: str, dst: str) -> None:
        if src not in sd:
            raise KeyError(f"missing expected tensor: {src}")
        _add(writer, dst, sd[src])
        used.add(src)

    for src, dst in TOP_TABLE:
        emit(src, dst)
    for i in range(n_layers):
        for s_suf, d_suf in BLOCK_TABLE:
            emit(f"model.audio_tower.layers.{i}.{s_suf}", f"enc.blocks.{i}.{d_suf}")

    conv = sd[UPSAMPLE_WEIGHT]  # [out = d_head * sub, in = d_head, k = 3]
    if conv.shape[2] != 3:
        raise ValueError(f"expected a 3-tap upsampler, got {conv.shape}")
    for k in range(3):
        _add(writer, f"diar.upsample.tap{k}.weight", np.ascontiguousarray(conv[:, :, k]))
    used.add(UPSAMPLE_WEIGHT)

    unexpected = sorted(set(sd) - used)
    if unexpected:
        raise ValueError(f"{len(unexpected)} unmapped tensors, e.g. {unexpected[:8]}")
    print(f"Emitted {len(used)} source tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote GGUF: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="HF repo id or a local directory with model.safetensors + configs")
    ap.add_argument("--repo-id", default=None, help="HF repo id (provenance)")
    ap.add_argument("--out", default=None, help="Override output GGUF path")
    args = ap.parse_args()

    repo_id = args.repo_id or (args.model if not Path(args.model).exists() else None)
    out_path = Path(args.out) if args.out else Path("models") / VARIANT / gguf_name(VARIANT, REFERENCE_DTYPE_LABEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(args.model, out_path, repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
