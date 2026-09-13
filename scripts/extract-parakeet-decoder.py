# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf>=0.17"]
# ///
"""Extract the TDT V3 decoder for use with its original GGUF's Core ML encoder."""
import argparse
import hashlib
from pathlib import Path

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def extract(source: Path, output: Path):
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from source")
    reader = GGUFReader(str(source))
    if reader.fields["stt.variant"].contents() != "tdt-0.6b-v3":
        raise ValueError("Only Parakeet TDT 0.6B V3 is supported")
    if not any(t.name.startswith("enc.") for t in reader.tensors):
        raise ValueError("Source must contain the original encoder")
    decoder = [t for t in reader.tensors if not t.name.startswith("enc.")]
    if not decoder or any(not t.name.startswith(("pred.", "joint.")) for t in decoder):
        raise ValueError("Unexpected decoder tensors")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output), "parakeet")
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in {
            "general.architecture", "stt.parakeet.decoder_only",
            "stt.parakeet.source_sha256",
        }:
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), kind, sub_type=subtype)
    writer.add_bool("stt.parakeet.decoder_only", True)
    with source.open("rb") as stream:
        writer.add_string("stt.parakeet.source_sha256", hashlib.file_digest(stream, "sha256").hexdigest())
    for tensor in decoder:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {len(decoder)} decoder tensors ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)
