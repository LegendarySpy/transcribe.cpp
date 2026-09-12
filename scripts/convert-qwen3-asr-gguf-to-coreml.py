# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["coremltools==9.0", "gguf>=0.17", "numpy>=1.26"]
# ///
"""Export a Qwen3-ASR audio encoder from transcribe.cpp GGUF weights.

The graph mirrors src/arch/qwen3_asr/encoder.cpp: the mel is cut into
n_window*2-frame chunks, each chunk runs the 3x Conv2d subsampler and gets
the same sinusoidal positions, the chunks are flattened into one sequence,
and the bidirectional blocks plus the LN/proj head run over it. The valid
rows are a prefix of that sequence ((n_chunks-1) rows per full chunk plus
aftercnn(last) rows), so mel_length masks the rest out of the attention
keys and the adapter reads the prefix.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import coremltools as ct
import gguf
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def aftercnn(n):
    for _ in range(3):
        n = (n + 1) // 2
    return n


def convert(source: Path, output: Path, max_frames: int):
    r = gguf.GGUFReader(str(source))
    if r.fields["general.architecture"].contents() != "qwen3_asr":
        raise ValueError("Expected a transcribe.cpp Qwen3-ASR GGUF")
    prefix = "stt.qwen3_asr.encoder."
    meta = {
        k: f.contents()
        for k, f in r.fields.items()
        if k.startswith((prefix, "stt.variant"))
    }
    d = int(meta[prefix + "d_model"])
    heads = int(meta[prefix + "n_heads"])
    layers = int(meta[prefix + "n_layers"])
    mels = int(meta[prefix + "num_mel_bins"])
    ds_h = int(meta[prefix + "downsample_hidden"])
    out_dim = int(meta[prefix + "output_dim"])
    chunk = 2 * int(meta[prefix + "n_window"])
    if meta[prefix + "activation"] != "gelu":
        raise ValueError("Expected a GELU encoder")
    if max_frames % chunk != 0:
        raise ValueError(f"--max-frames must be a multiple of {chunk}")
    n_chunks = max_frames // chunk
    rows = aftercnn(chunk)
    f_ds = aftercnn(mels)
    T = n_chunks * rows
    weights = {t.name: t for t in r.tensors if t.name.startswith("enc.")}

    def w(name):
        t = weights[name]
        return np.ascontiguousarray(
            gguf.dequantize(t.data, t.tensor_type).reshape(
                tuple(int(v) for v in t.shape[::-1])
            ),
            dtype=np.float32,
        )

    def lin(x, name):
        args = {"x": x, "weight": w(name + ".weight")}
        if name + ".bias" in weights:
            args["bias"] = w(name + ".bias")
        return mb.linear(**args)

    def norm(x, name):
        return mb.layer_norm(
            x=x,
            axes=[-1],
            gamma=w(name + ".weight"),
            beta=w(name + ".bias"),
            epsilon=1e-5,
        )

    def split_heads(x):
        return mb.transpose(
            x=mb.reshape(x=x, shape=[1, T, heads, d // heads]), perm=[0, 2, 1, 3]
        )

    # SinusoidsPositionEmbedding over one chunk's rows: sin block then cos block.
    half = d // 2
    inv_ts = np.exp(-np.log(10000.0) / (half - 1) * np.arange(half))
    scaled = np.arange(rows)[:, None] * inv_ts[None, :]
    pe = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, mels, max_frames)),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ],
        opset_version=ct.target.macOS13,
    )
    def program(logmel_data, mel_length):
        x = mb.reshape(x=logmel_data, shape=[mels, n_chunks, chunk])
        x = mb.expand_dims(x=mb.transpose(x=x, perm=[1, 0, 2]), axes=[1])
        for i in range(3):
            x = mb.conv(
                x=x,
                weight=w(f"enc.conv.{i}.weight"),
                bias=w(f"enc.conv.{i}.bias"),
                strides=[2, 2],
                pad_type="custom",
                pad=[1, 1, 1, 1],
            )
            x = mb.gelu(x=x, mode="EXACT")
        # [n_chunks, ds_h, f_ds, rows] -> [n_chunks, rows, ds_h * f_ds]
        x = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 1, 2]), shape=[n_chunks, rows, ds_h * f_ds])
        x = mb.add(x=lin(x, "enc.conv_out"), y=pe[None, :, :])
        x = mb.reshape(x=x, shape=[1, T, d])

        # Valid rows: full chunks contribute `rows` each, the last one
        # aftercnn(last_real_mel).
        one = np.array([1], np.int32)
        full = mb.floor_div(x=mb.sub(x=mel_length, y=one), y=np.array([chunk], np.int32))
        last = mb.sub(x=mel_length, y=mb.mul(x=full, y=np.array([chunk], np.int32)))
        for _ in range(3):
            last = mb.floor_div(x=mb.add(x=last, y=one), y=np.array([2], np.int32))
        valid_rows = mb.add(x=mb.mul(x=full, y=np.array([rows], np.int32)), y=last)
        key_mask = mb.reshape(
            x=mb.select(
                cond=mb.less(x=np.arange(T, dtype=np.int32), y=valid_rows),
                a=np.zeros(T, np.float32),
                b=np.full(T, -10000, np.float32),
            ),
            shape=[1, 1, 1, T],
        )
        for i in range(layers):
            name = f"enc.blocks.{i}"
            h = norm(x, name + ".norm_attn")
            q, k, v = [split_heads(lin(h, name + ".attn." + kind)) for kind in "qkv"]
            scores = mb.add(
                x=mb.mul(x=mb.matmul(x=q, y=k, transpose_y=True), y=np.float32((d // heads) ** -0.5)),
                y=key_mask,
            )
            attended = mb.matmul(x=mb.softmax(x=scores, axis=-1), y=v)
            attended = mb.reshape(x=mb.transpose(x=attended, perm=[0, 2, 1, 3]), shape=[1, T, d])
            x = mb.add(x=x, y=lin(attended, name + ".attn.out"))
            y = mb.gelu(x=lin(norm(x, name + ".norm_ffn"), name + ".ffn.fc1"), mode="EXACT")
            x = mb.add(x=x, y=lin(y, name + ".ffn.fc2"))
        x = norm(x, "enc.ln_post")
        x = lin(mb.gelu(x=lin(x, "enc.proj1"), mode="EXACT"), "enc.proj2")
        return mb.identity(x=x, name="output")

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    model.user_defined_metadata["transcribe.variant"] = str(meta["stt.variant"])
    with source.open("rb") as f:
        model.user_defined_metadata["transcribe.gguf.sha256"] = hashlib.file_digest(
            f, "sha256"
        ).hexdigest()
    model.user_defined_metadata["transcribe.gguf.filename"] = source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print("Saved", output, "output rows", T, "of", out_dim, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1500,
        help="Mel-frame capacity, a multiple of the 2*n_window chunk; 1500 covers 15 seconds",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the package with Xcode's coremlcompiler",
    )
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage":
        parser.error("--output must end in .mlpackage")
    convert(args.gguf, args.output, args.max_frames)
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
