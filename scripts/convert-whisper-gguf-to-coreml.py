# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["coremltools==9.0", "gguf>=0.17", "numpy>=1.26"]
# ///
"""Build a Whisper Core ML encoder for the Apple Neural Engine.

Reads the encoder from a transcribe.cpp GGUF or a whisper.cpp ggml .bin
(F16/F32/quantized). No reference checkpoints are downloaded; quantized
tensors are expanded and converted to FP16. The decoder and frontend remain
in transcribe.cpp.

The graph follows Apple's ANE transformer layout: activations stay (1, C, 1, S),
linears are 1x1 convs, LayerNorm runs on the channel axis, and attention is
split per head and per query block so every score tensor stays small. Large
encoders are split into chained programs (a Core ML pipeline) because the ANE
compiler rejects one 1.2 GB program and silently runs it on the CPU. One
encoder serves every quantization of a variant; export it from the F16 file.

uv run --python 3.11 scripts/convert-whisper-gguf-to-coreml.py model.gguf \
    --output models/model-encoder.mlpackage --compile
"""

import argparse
import hashlib
import struct
import subprocess
from pathlib import Path

import coremltools as ct
import gguf
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.optimize import coreml as cto

# Stored weight bytes per pipeline stage. 16 large-v3 layers (630 MB of FP16)
# compile for the ANE; all 32 in one program fall back to the CPU.
STAGE_BYTES = 640_000_000

# The ANE flushes FP16 products below 2^-14 to zero, which erases most of
# fc2's small terms. The residual stream is carried scaled per channel by a
# power of two (up to 64, keeping peaks under RESIDUAL_BUDGET) and V by 64.
RESIDUAL_BUDGET = 1024.0
MAX_SCALE = 64.0

BIN_NAMES = {
    "encoder.positional_embedding": "enc.pos_emb.weight",
    "encoder.conv1.weight": "enc.conv.0.weight",
    "encoder.conv1.bias": "enc.conv.0.bias",
    "encoder.conv2.weight": "enc.conv.1.weight",
    "encoder.conv2.bias": "enc.conv.1.bias",
    "encoder.ln_post.weight": "enc.final_norm.weight",
    "encoder.ln_post.bias": "enc.final_norm.bias",
}
BIN_BLOCK_NAMES = {
    "attn_ln": "norm_attn",
    "attn.query": "attn.q",
    "attn.key": "attn.k",
    "attn.value": "attn.v",
    "attn.out": "attn.out",
    "mlp_ln": "norm_ffn",
    "mlp.0": "ffn.fc1",
    "mlp.2": "ffn.fc2",
}


def bin_variant(n_vocab, n_audio_layer, n_text_layer, n_mels):
    # Mirrors detect_variant() in src/arch/whisper/bin_load.cpp.
    names = {
        4: "whisper-tiny",
        6: "whisper-base",
        12: "whisper-small",
        24: "whisper-medium",
    }
    base = names.get(n_audio_layer, "whisper")
    if n_audio_layer == 32:
        if n_text_layer == 4:
            base = "whisper-large-v3-turbo"
        else:
            base = "whisper-large-v3" if n_mels == 128 else "whisper-large"
    return base if n_vocab >= 51865 else base + ".en"


def read_bin(source: Path):
    data = np.memmap(source, dtype=np.uint8, mode="r")
    pos = 0

    def ints(n):
        nonlocal pos
        values = struct.unpack_from(f"<{n}i", data, pos)
        pos += 4 * n
        return values

    if ints(1)[0] != 0x67676D6C:
        raise ValueError("Expected a GGUF or whisper.cpp ggml .bin")
    n_vocab, n_ctx, width, heads, layers, _, _, _, text_layers, n_mels, _ = ints(11)
    n_mel, n_fft = ints(2)
    pos += 4 * n_mel * n_fft
    for _ in range(ints(1)[0]):
        length = ints(1)[0]
        pos += length
    tensors = {}
    while pos < len(data):
        n_dims, name_len, ttype = ints(3)
        ne = ints(n_dims)
        name = bytes(data[pos : pos + name_len]).decode()
        pos += name_len
        qtype = gguf.GGMLQuantizationType(ttype)
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        nbytes = int(np.prod(ne)) // block * size
        if name.startswith("encoder."):
            if name in BIN_NAMES:
                name = BIN_NAMES[name]
            else:  # encoder.blocks.N.<module>.<param>
                parts = name.split(".")
                module = BIN_BLOCK_NAMES[".".join(parts[3:-1])]
                name = f"enc.blocks.{parts[2]}.{module}.{parts[-1]}"
            tensors[name] = (data[pos : pos + nbytes], qtype, tuple(reversed(ne)))
        pos += nbytes
    hp = {
        "d_model": width,
        "max_source_positions": n_ctx,
        "n_heads": heads,
        "num_mel_bins": n_mels,
        "n_layers": layers,
        "activation": "gelu",
    }
    return hp, tensors, bin_variant(n_vocab, layers, text_layers, n_mels)


def read_gguf(source: Path):
    reader = gguf.GGUFReader(str(source))
    if reader.fields["general.architecture"].contents() != "whisper":
        raise ValueError("Expected a transcribe.cpp Whisper GGUF")
    prefix = "stt.whisper.encoder."
    hp = {
        key[len(prefix) :]: field.contents()
        for key, field in reader.fields.items()
        if key.startswith(prefix)
    }
    tensors = {
        t.name: (t.data, t.tensor_type, tuple(int(n) for n in t.shape[::-1]))
        for t in reader.tensors
        if t.name.startswith("enc.")
    }
    return hp, tensors, str(reader.fields["stt.variant"].contents())


def read_source(source: Path):
    with source.open("rb") as handle:
        magic = handle.read(4)
    return read_gguf(source) if magic == b"GGUF" else read_bin(source)


def residual_scales(hp, tensors, weight):
    """Per-channel residual scales from an FP32 NumPy pass over synthetic input.

    Whisper's residual stream has a few outlier channels (up to ~2700 in
    medium) whose positions and sizes barely depend on the input. Half noise,
    half silence finds them; real speech stays within ~3x of these peaks.
    """
    width, frames, heads = (
        int(hp[key]) for key in ("d_model", "max_source_positions", "n_heads")
    )
    n_mels, layers = int(hp["num_mel_bins"]), int(hp["n_layers"])

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def conv(x, i, stride):
        w, b = weight(f"enc.conv.{i}.weight"), weight(f"enc.conv.{i}.bias")
        x = np.pad(x, ((0, 0), (1, 1)))
        n = (x.shape[1] - 3) // stride + 1
        return (
            sum(w[:, :, k] @ x[:, k : k + stride * n : stride] for k in range(3))
            + b[:, None]
        )

    def norm(x, name):
        x = x - x.mean(-1, keepdims=True)
        x = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)
        return x * weight(name + ".weight") + weight(name + ".bias")

    def linear(x, name):
        y = x @ weight(name + ".weight").T
        return y + weight(name + ".bias") if name + ".bias" in tensors else y

    mel = np.full((n_mels, 2 * frames), -1.5, dtype=np.float32)
    mel[:, :frames] = np.random.default_rng(0).normal(0, 0.5, (n_mels, frames))
    x = gelu(conv(gelu(conv(mel, 0, 1)), 1, 2)).T + weight("enc.pos_emb.weight")
    peak = np.abs(x).max(0)
    for i in range(layers):
        name = f"enc.blocks.{i}"
        h = norm(x, name + ".norm_attn")
        q, k, v = (
            linear(h, f"{name}.attn.{t}").reshape(frames, heads, -1).transpose(1, 0, 2)
            for t in "qkv"
        )
        scores = q @ k.transpose(0, 2, 1) * (width // heads) ** -0.5
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        scores /= scores.sum(-1, keepdims=True)
        attended = (scores @ v).transpose(1, 0, 2).reshape(frames, width)
        x = x + linear(attended, name + ".attn.out")
        peak = np.maximum(peak, np.abs(x).max(0))
        hidden = gelu(linear(norm(x, name + ".norm_ffn"), name + ".ffn.fc1"))
        x = x + linear(hidden, name + ".ffn.fc2")
        peak = np.maximum(peak, np.abs(x).max(0))
    scale = 2.0 ** np.floor(np.log2(RESIDUAL_BUDGET / np.maximum(peak, 1e-3)))
    return np.clip(scale, 1.0, MAX_SCALE).astype(np.float32)


def build_stages(hp, tensors, stages, query_blocks):
    width, frames, heads = (
        int(hp[key]) for key in ("d_model", "max_source_positions", "n_heads")
    )
    n_mels, layers = int(hp["num_mel_bins"]), int(hp["n_layers"])
    if (
        min(width, frames, heads, n_mels, layers) <= 0
        or width % heads
        or hp["activation"] != "gelu"
    ):
        raise ValueError("Unsupported Whisper encoder configuration")
    dh = width // heads

    def weight(name):
        raw, qtype, shape = tensors[name]
        w = gguf.dequantize(raw, qtype).reshape(shape)
        # whisper.cpp stores conv biases as [1, d_model].
        return np.ascontiguousarray(
            w.reshape(-1) if name.endswith(".bias") else w, dtype=np.float32
        )

    scale = residual_scales(hp, tensors, weight)
    values, counts = np.unique(scale.astype(int), return_counts=True)
    print("Residual channel scales:", dict(zip(values.tolist(), counts.tolist())))
    channel = scale.reshape(1, width, 1, 1)

    def linear(x, name, rows=1.0, bias=None):
        # rows scales output channels; bias defaults to the same scale.
        w = weight(name + ".weight") * np.reshape(rows, (-1, 1))
        args = {"x": x, "weight": w.reshape(*w.shape, 1, 1)}
        if name + ".bias" in tensors:
            args["bias"] = weight(name + ".bias") * (rows if bias is None else bias)
        return mb.conv(**args)

    def norm(x, name):
        return mb.layer_norm(
            x=mb.mul(x=x, y=1 / channel),
            axes=[1],
            gamma=weight(name + ".weight"),
            beta=weight(name + ".bias"),
            epsilon=1e-5,
        )

    def head(x, h, axis, begin=0, end=frames):
        lo, hi = [0, 0, 0, 0], [1, frames, 1, frames]
        lo[axis], hi[axis] = h * dh, (h + 1) * dh
        lo[4 - axis], hi[4 - axis] = begin, end
        return mb.slice_by_index(x=x, begin=lo, end=hi)

    blocks = [round(frames * b / query_blocks) for b in range(query_blocks + 1)]

    def block(x, i):
        name = f"enc.blocks.{i}"
        h = norm(x, name + ".norm_attn")
        q = linear(h, name + ".attn.q", np.float32(dh**-0.5))
        k = mb.transpose(x=linear(h, name + ".attn.k"), perm=[0, 3, 2, 1])
        v = linear(h, name + ".attn.v", np.float32(MAX_SCALE))
        rows = []
        for b in range(query_blocks):
            cols = []
            for j in range(heads):
                # (1, S, 1, dh) x (1, dh, 1, Sq) -> (1, S, 1, Sq), keys on axis 1.
                qh = head(q, j, 1, blocks[b], blocks[b + 1])
                scores = mb.einsum(
                    values=(head(k, j, 3), qh), equation="nchw,nwhu->nchu"
                )
                weights = mb.softmax(x=scores, axis=1)
                cols.append(
                    mb.einsum(
                        values=(head(v, j, 1), weights), equation="nchw,nwhu->nchu"
                    )
                )
            rows.append(mb.concat(values=cols, axis=1))
        attended = rows[0] if query_blocks == 1 else mb.concat(values=rows, axis=3)
        x = mb.add(
            x=x, y=linear(attended, name + ".attn.out", scale / MAX_SCALE, scale)
        )
        hidden = linear(norm(x, name + ".norm_ffn"), name + ".ffn.fc1")
        hidden = mb.gelu(x=hidden, mode="EXACT")
        return mb.add(x=x, y=linear(hidden, name + ".ffn.fc2", scale))

    bounds = [round(layers * s / stages) for s in range(stages + 1)]
    programs = []
    for s in range(stages):
        first, last = s == 0, s == stages - 1
        if first:
            spec = mb.TensorSpec(shape=(1, n_mels, frames * 2))
        else:
            spec = mb.TensorSpec(shape=(1, width, 1, frames), dtype=types.fp16)

        @mb.program(input_specs=[spec], opset_version=ct.target.macOS14)
        def program(x):
            if first:
                x = mb.expand_dims(x=x, axes=[2])
                for i, stride in enumerate([1, 2]):
                    w = weight(f"enc.conv.{i}.weight")
                    x = mb.conv(
                        x=x,
                        weight=w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]),
                        bias=weight(f"enc.conv.{i}.bias"),
                        strides=[1, stride],
                        pad_type="custom",
                        pad=[0, 0, 1, 1],
                    )
                    x = mb.gelu(x=x, mode="EXACT")
                pos = weight("enc.pos_emb.weight").T
                x = mb.add(x=x, y=pos.reshape(1, width, 1, frames))
                x = mb.mul(x=x, y=channel)
            else:
                x = mb.cast(x=x, dtype="fp32")
            for i in range(bounds[s], bounds[s + 1]):
                x = block(x, i)
            if last:
                x = norm(x, "enc.final_norm")
                x = mb.transpose(x=mb.squeeze(x=x, axes=[2]), perm=[0, 2, 1])
                return mb.identity(x=mb.cast(x=x, dtype="fp32"), name="output")
            return mb.identity(x=mb.cast(x=x, dtype="fp16"), name=f"hidden_{s}")

        programs.append(program)
    return programs


def convert(source, output, stages, query_blocks, int8, also_variants):
    hp, tensors, variant = read_source(source)
    if stages <= 0:
        # 12 d^2 weights per block.
        layer_bytes = (12 if int8 else 24) * int(hp["d_model"]) ** 2
        stages = -(-layer_bytes * int(hp["n_layers"]) // STAGE_BYTES)
    models = []
    for s, program in enumerate(build_stages(hp, tensors, stages, query_blocks)):
        last = s == stages - 1
        model = ct.convert(
            program,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
            inputs=[
                ct.TensorType(name="x", dtype=np.float32 if s == 0 else np.float16)
            ],
            outputs=[
                ct.TensorType(
                    name="output" if last else f"hidden_{s}",
                    dtype=np.float32 if last else np.float16,
                )
            ],
            skip_model_load=True,
        )
        if int8:
            config = cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", granularity="per_channel"
            )
            model = cto.linear_quantize_weights(
                model, cto.OptimizationConfig(global_config=config)
            )
        spec = model.get_spec()
        ct.utils.rename_feature(spec, "x", f"hidden_{s - 1}" if s else "logmel_data")
        models.append(
            ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
        )
        print(f"Built stage {s + 1}/{stages}", flush=True)
    model = models[0] if stages == 1 else ct.utils.make_pipeline(*models)
    with source.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    model.user_defined_metadata["transcribe.gguf.sha256"] = checksum
    model.user_defined_metadata["transcribe.gguf.filename"] = source.name
    variant = ",".join([variant, *also_variants])
    model.user_defined_metadata["transcribe.variant"] = variant
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"Saved {output} ({variant}, source SHA256 {checksum})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="transcribe.cpp GGUF or whisper.cpp .bin"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the package with Xcode's coremlcompiler",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=0,
        help="Chained Core ML programs (default: by encoder size)",
    )
    parser.add_argument(
        "--query-blocks",
        type=int,
        default=4,
        help="Split attention queries into this many blocks per head",
    )
    parser.add_argument(
        "--also-variant",
        action="append",
        default=[],
        help="Another variant sharing this encoder (e.g. distil-large-v3.5)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Store weights as per-channel int8 (computation stays FP16)",
    )
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage":
        parser.error("--output must end in .mlpackage")
    if args.query_blocks < 1:
        parser.error("--query-blocks must be at least 1")
    convert(
        args.source,
        args.output,
        args.stages,
        args.query_blocks,
        args.int8,
        args.also_variant,
    )
    if args.compile:
        subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                str(args.output),
                str(args.output.parent),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
