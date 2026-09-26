# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["coremltools==9.0", "gguf>=0.17", "numpy>=1.26"]
# ///
"""Build a Whisper Core ML encoder directly from a transcribe.cpp GGUF.

No reference checkpoints are downloaded. Quantized encoder tensors are
expanded from the GGUF and converted to Core ML's FP16 compute precision.
The decoder and frontend remain in transcribe.cpp.

uv run --python 3.11 scripts/convert-whisper-gguf-to-coreml.py model.gguf \
    --output models/model-encoder.mlpackage --compile
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

import coremltools as ct
import gguf
import numpy as np
from coremltools.converters.mil import Builder as mb


def convert(source: Path, output: Path):
    reader = gguf.GGUFReader(str(source))
    arch = reader.fields["general.architecture"].contents()
    if arch != "whisper":
        raise ValueError("Expected a transcribe.cpp Whisper GGUF")
    prefix = "stt.whisper.encoder."
    hp = {
        key[len(prefix) :]: field.contents()
        for key, field in reader.fields.items()
        if key.startswith(prefix)
    }
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
    tensors = {
        tensor.name: tensor
        for tensor in reader.tensors
        if tensor.name.startswith("enc.")
    }
    def weight(name):
        tensor = tensors[name]
        return np.ascontiguousarray(
            gguf.dequantize(tensor.data, tensor.tensor_type).reshape(
                tuple(int(n) for n in tensor.shape[::-1])
            ),
            dtype=np.float32,
        )

    def linear(x, name):
        args = {"x": x, "weight": weight(name + ".weight")}
        if name + ".bias" in tensors:
            args["bias"] = weight(name + ".bias")
        return mb.linear(**args)

    def norm(x, name):
        return mb.layer_norm(
            x=x,
            axes=[-1],
            gamma=weight(name + ".weight"),
            beta=weight(name + ".bias"),
            epsilon=1e-5,
        )

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, n_mels, frames * 2))],
        opset_version=ct.target.macOS13,
    )
    def program(logmel_data):
        x = logmel_data
        for i, stride in enumerate([1, 2]):
            x = mb.conv(
                x=x,
                weight=weight(f"enc.conv.{i}.weight"),
                bias=weight(f"enc.conv.{i}.bias"),
                strides=[stride],
                pad_type="custom",
                pad=[1, 1],
            )
            x = mb.gelu(x=x, mode="EXACT")
        x = mb.transpose(x=x, perm=[0, 2, 1])
        x = mb.add(x=x, y=weight("enc.pos_emb.weight"))
        for i in range(layers):
            name = f"enc.blocks.{i}"
            normalized = norm(x, name + ".norm_attn")
            projections = []
            for kind in ["q", "k", "v"]:
                p = linear(normalized, name + ".attn." + kind)
                p = mb.reshape(x=p, shape=[1, frames, heads, width // heads])
                projections.append(mb.transpose(x=p, perm=[0, 2, 1, 3]))
            q, k, v = projections
            q = mb.mul(x=q, y=np.float32((width // heads) ** -0.5))
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            attention = mb.softmax(x=scores, axis=-1)
            attended = mb.matmul(x=attention, y=v)
            attended = mb.transpose(x=attended, perm=[0, 2, 1, 3])
            attended = mb.reshape(x=attended, shape=[1, frames, width])
            x = mb.add(x=x, y=linear(attended, name + ".attn.out"))
            hidden = linear(norm(x, name + ".norm_ffn"), name + ".ffn.fc1")
            hidden = mb.gelu(x=hidden, mode="EXACT")
            x = mb.add(x=x, y=linear(hidden, name + ".ffn.fc2"))
        x = norm(x, "enc.final_norm")
        return mb.identity(x=x, name="output")

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    with source.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    model.user_defined_metadata["transcribe.gguf.sha256"] = checksum
    model.user_defined_metadata["transcribe.gguf.filename"] = source.name
    model.user_defined_metadata["transcribe.variant"] = str(
        reader.fields["stt.variant"].contents()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"Saved {output} (source SHA256 {checksum})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the package with Xcode's coremlcompiler",
    )
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage":
        parser.error("--output must end in .mlpackage")
    convert(args.gguf, args.output)
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
