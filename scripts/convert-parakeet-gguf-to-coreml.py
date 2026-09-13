# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["coremltools==9.0", "gguf>=0.17", "numpy>=1.26"]
# ///
"""Export a Parakeet or Canary FastConformer encoder from transcribe.cpp GGUF weights.

Full-context, batch-norm FastConformer encoders with 8x depthwise-striding
subsampling: the offline Parakeet variants and every Canary variant. Streaming
(chunked-attention) and speaker-kernel checkpoints are rejected.
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


def convert(source: Path, output: Path, max_frames: int):
    r = gguf.GGUFReader(str(source))
    arch = r.fields["general.architecture"].contents()
    if arch not in ("parakeet", "canary"):
        raise ValueError("Expected a transcribe.cpp Parakeet or Canary GGUF")
    prefix = f"stt.{arch}.encoder."
    meta = {
        k: f.contents()
        for k, f in r.fields.items()
        if k.startswith((prefix, "stt.frontend.", "stt.variant", "stt.canary.decoder."))
    }
    d = int(meta[prefix + "d_model"])
    heads = int(meta[prefix + "n_heads"])
    layers = int(meta[prefix + "n_layers"])
    mels = int(meta["stt.frontend.num_mels"])
    channels = int(meta[prefix + "subsampling_channels"])
    if (
        int(meta[prefix + "subsampling_factor"]) != 8
        or meta.get(prefix + "conv_norm_type", "batch_norm") != "batch_norm"
        or int(meta.get(prefix + "att_context_left", -1)) != -1
        or int(meta.get(prefix + "att_context_right", -1)) != -1
        or prefix + "spk_kernel_layers" in meta
        or d % heads != 0
    ):
        raise ValueError(
            "Expected a full-context, batch-norm, 8x-subsampling encoder without speaker kernels"
        )
    T = (max_frames + 7) // 8
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

    def silu(x):
        return mb.mul(x=x, y=mb.sigmoid(x=x))

    def conv(x, name, strides, pad, groups=1, weight=None, bias=None):
        args = {
            "x": x,
            "weight": w(name + ".weight") if weight is None else weight,
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "groups": groups,
        }
        if bias is not None:
            args["bias"] = bias
        elif name + ".bias" in weights:
            args["bias"] = w(name + ".bias")
        return mb.conv(**args)

    def split_heads(x, length):
        return mb.transpose(
            x=mb.reshape(x=x, shape=[1, length, heads, d // heads]), perm=[0, 2, 1, 3]
        )

    pos = np.arange(T - 1, -T, -1, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, d, 2, dtype=np.float32) * np.float32(-np.log(10000) / d))[
        None, :
    ]
    pe = np.empty((2 * T - 1, d), np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, mels, max_frames)),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
        ],
        opset_version=ct.target.macOS13,
    )
    def program(logmel_data, mel_length):
        length = mel_length
        x = mb.expand_dims(x=mb.transpose(x=logmel_data, perm=[0, 2, 1]), axes=[1])
        for a, b in [(0, None), (2, 3), (5, 6)]:
            x = conv(
                x,
                f"enc.pre_encode.conv.{a}",
                [2, 2],
                [1, 1, 1, 1],
                groups=1 if a == 0 else channels,
            )
            if b is not None:
                x = conv(x, f"enc.pre_encode.conv.{b}", [1, 1], [0, 0, 0, 0])
            x = mb.relu(x=x)
            length = mb.floor_div(
                x=mb.add(x=length, y=np.array([1], np.int32)), y=np.array([2], np.int32)
            )
            mask = mb.less(x=np.arange(x.shape[2], dtype=np.int32), y=length)
            x = mb.mul(
                x=x,
                y=mb.reshape(
                    x=mb.cast(x=mask, dtype="fp32"), shape=[1, 1, x.shape[2], 1]
                ),
            )
        x = mb.reshape(
            x=mb.transpose(x=x, perm=[0, 2, 1, 3]),
            shape=[1, T, channels * ((mels + 7) // 8)],
        )
        x = lin(x, "enc.pre_encode.out")
        # xscaling multiplies this stream by sqrt(d_model). Inside block 0 it is
        # consumed only through LayerNorms and the block ends in norm_out, so
        # scaling the block-0 sublayer outputs by 1/sqrt(d_model) instead is
        # equivalent up to LayerNorm eps and keeps the residual stream inside
        # FP16 range: pre_encode outputs reach 5e3 on the 80-mel checkpoints.
        xscale = np.float32(d**-0.5) if meta.get(prefix + "xscaling", False) else None
        valid = mb.less(x=np.arange(T, dtype=np.int32), y=length)
        key_mask = mb.reshape(
            x=mb.select(
                cond=valid, a=np.zeros(T, np.float32), b=np.full(T, -10000, np.float32)
            ),
            shape=[1, 1, 1, T],
        )
        conv_mask = mb.reshape(x=mb.cast(x=valid, dtype="fp32"), shape=[1, 1, T])
        for i in range(layers):
            name = f"enc.blocks.{i}"

            def residual(y, weight=1.0):
                if xscale is not None and i == 0:
                    weight = weight * xscale
                return mb.mul(x=y, y=np.float32(weight)) if weight != 1.0 else y

            y = lin(
                silu(lin(norm(x, name + ".norm_ff1"), name + ".ff1.linear1")),
                name + ".ff1.linear2",
            )
            x = mb.add(x=x, y=residual(y, 0.5))
            normalized = norm(x, name + ".norm_attn")
            q, k, v = [
                split_heads(lin(normalized, name + ".attn.linear_" + kind), T)
                for kind in ["q", "k", "v"]
            ]
            p = split_heads(
                lin(mb.const(val=pe[None, :, :]), name + ".attn.linear_pos"), 2 * T - 1
            )
            qu = mb.add(
                x=q, y=w(name + ".attn.pos_bias_u").reshape(1, heads, 1, d // heads)
            )
            qv = mb.add(
                x=q, y=w(name + ".attn.pos_bias_v").reshape(1, heads, 1, d // heads)
            )
            ac = mb.matmul(x=qu, y=k, transpose_y=True)
            bd = mb.matmul(x=qv, y=p, transpose_y=True)
            # Relative-position shift matches the GGUF encoder attention graph.
            bd = mb.pad(x=bd, pad=[0, 0, 0, 0, 0, 0, 1, 0])
            bd = mb.reshape(x=bd, shape=[1, heads, 2 * T, T])
            bd = mb.slice_by_index(x=bd, begin=[0, 0, 1, 0], end=[1, heads, 2 * T, T])
            bd = mb.reshape(x=bd, shape=[1, heads, T, 2 * T - 1])
            bd = mb.slice_by_index(x=bd, begin=[0, 0, 0, 0], end=[1, heads, T, T])
            scores = mb.add(
                x=mb.mul(x=mb.add(x=ac, y=bd), y=np.float32((d // heads) ** -0.5)),
                y=key_mask,
            )
            attended = mb.matmul(x=mb.softmax(x=scores, axis=-1), y=v)
            attended = mb.reshape(
                x=mb.transpose(x=attended, perm=[0, 2, 1, 3]), shape=[1, T, d]
            )
            x = mb.add(x=x, y=residual(lin(attended, name + ".attn.linear_out")))
            y = mb.transpose(x=norm(x, name + ".norm_conv"), perm=[0, 2, 1])
            y = conv(y, name + ".conv.pointwise1", [1], [0, 0])
            gate, value = mb.split(x=y, num_splits=2, axis=1)
            y = mb.mul(x=mb.mul(x=gate, y=mb.sigmoid(x=value)), y=conv_mask)
            dw = w(name + ".conv.depthwise.weight").reshape(d, 1, -1)
            pad = (dw.shape[2] - 1) // 2
            y = conv(y, name + ".conv.depthwise", [1], [pad, pad], groups=d, weight=dw)
            scale = w(name + ".conv.bn.weight") / np.sqrt(
                w(name + ".conv.bn.running_var") + np.float32(1e-5)
            )
            bias = w(name + ".conv.bn.bias") - w(name + ".conv.bn.running_mean") * scale
            y = mb.add(x=mb.mul(x=y, y=scale.reshape(1, d, 1)), y=bias.reshape(1, d, 1))
            y = conv(silu(y), name + ".conv.pointwise2", [1], [0, 0])
            x = mb.add(x=x, y=residual(mb.transpose(x=y, perm=[0, 2, 1])))
            y = lin(
                silu(lin(norm(x, name + ".norm_ff2"), name + ".ff2.linear1")),
                name + ".ff2.linear2",
            )
            x = norm(mb.add(x=x, y=residual(y, 0.5)), name + ".norm_out")
        if meta.get("stt.canary.decoder.encoder_decoder_proj", False):
            x = lin(x, "enc.proj")
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
    print("Saved", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1501,
        help="Encoder mel-frame capacity; 1501 covers about 15 seconds",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the package with Xcode's coremlcompiler",
    )
    args = parser.parse_args()
    if args.output.suffix != ".mlpackage":
        parser.error("--output must end in .mlpackage")
    if args.max_frames < 9:
        parser.error("--max-frames must be at least 9")
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
