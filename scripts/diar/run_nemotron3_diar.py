#!/usr/bin/env python3
"""
run_nemotron3_diar.py - Nemotron-3 Diarization over an AMI-style manifest,
from either the HF Transformers reference or the C++ port.

Both sides save the raw [T, 8] sigmoid matrix (10 ms frames) per meeting and
turn it into an RTTM with the same rule, so only the probabilities differ:
a speaker is active on frames with p > --threshold, runs shorter than
--min-dur seconds are dropped. The defaults (0.5, no minimum) are the plain
thresholding the model card's `logits.sigmoid()` example implies.

  # reference (Transformers)
  uv run --project scripts/envs/nemotron3_diar scripts/diar/run_nemotron3_diar.py ref \
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl --preset offline \
    --probs-dir reports/diar/probs/nemotron-3-diarization-REF-offline \
    --pred-dir reports/diar/pred/nemotron-3-diarization-REF-offline

  # C++ port
  uv run --project scripts/envs/nemotron3_diar scripts/diar/run_nemotron3_diar.py cpp \
    --gguf models/nemotron-3-diarization/nemotron-3-diarization-F32.gguf --preset offline \
    --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
    --probs-dir reports/diar/probs/nemotron-3-diarization-CPP-F32-offline \
    --pred-dir reports/diar/pred/nemotron-3-diarization-CPP-F32-offline

  # C++ port, push-audio stream (transcribe_stream_*) fed in 100 ms pieces
  uv run --project scripts/envs/nemotron3_diar scripts/diar/run_nemotron3_diar.py cpp \
    --gguf models/nemotron-3-diarization/nemotron-3-diarization-Q8_0.gguf --preset low_latency \
    --stream-chunk-ms 100 --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
    --probs-dir reports/diar/probs/nemotron-3-diarization-CPP-Q8_0-metal-low_latency-stream \
    --pred-dir reports/diar/pred/nemotron-3-diarization-CPP-Q8_0-metal-low_latency-stream

  uv run scripts/diar/score_der.py --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl \
    --pred-dir reports/diar/pred/nemotron-3-diarization-CPP-F32-offline \
    --out reports/diar/nemotron-3-diarization-CPP-F32-offline.ami-ihm-test-fa.score.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FRAME_S = 0.01
N_SPK = 8


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def probs_to_rttm(probs: np.ndarray, uri: str, threshold: float, min_dur: float) -> str:
    lines = []
    active = probs > threshold
    for spk in range(probs.shape[1]):
        start = None
        for t in range(probs.shape[0] + 1):
            on = t < probs.shape[0] and active[t, spk]
            if on and start is None:
                start = t
            elif not on and start is not None:
                dur = (t - start) * FRAME_S
                if dur >= min_dur:
                    lines.append(
                        f"SPEAKER {uri} 1 {start * FRAME_S:.3f} {dur:.3f} <NA> <NA> speaker_{spk} <NA> <NA>"
                    )
                start = None
    return "\n".join(lines) + ("\n" if lines else "")


def run_ref(rows: list[dict], args: argparse.Namespace):
    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dump_reference_nemotron3_diar_transformers import _features, _load, run_preset

    processor, model = _load(args.model)
    for row in rows:
        audio, sr = sf.read(row["audio"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, row["audio"]
        feats, mask = _features(processor, audio)
        t0 = time.perf_counter()
        probs = run_preset(model, feats, mask, args.preset).numpy()
        yield row, probs, time.perf_counter() - t0, len(audio) / sr


def run_cpp(rows: list[dict], args: argparse.Namespace):
    cli = Path(args.cli)
    for row in rows:
        with tempfile.TemporaryDirectory() as dump_dir:
            env = dict(os.environ, TRANSCRIBE_DUMP_DIR=dump_dir, TRANSCRIBE_NEMOTRON3_DIAR_PRESET=args.preset)
            cmd = [str(cli), "-q", "-m", args.gguf, "--backend", args.backend, row["audio"]]
            if args.threads:
                cmd[1:1] = ["--threads", str(args.threads)]
            if args.stream_chunk_ms:
                cmd[1:1] = ["--stream-chunk-ms", str(args.stream_chunk_ms)]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                raise SystemExit(f"transcribe-cli failed on {row['id']}:\n{proc.stderr[-2000:]}")
            meta = json.loads((Path(dump_dir) / "diar.probs.json").read_text())
            probs = np.fromfile(Path(dump_dir) / "diar.probs.f32", dtype=np.float32).reshape(meta["shape"])
        yield row, probs, elapsed, float(row.get("duration", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("side", choices=["ref", "cpp"])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--preset", default="offline")
    ap.add_argument("--probs-dir", required=True, type=Path, help="Raw [T, 8] probs per meeting (.npy)")
    ap.add_argument("--pred-dir", required=True, type=Path, help="Predicted RTTM per meeting")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-dur", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="First N meetings only")
    ap.add_argument("--model", default="nvidia/Nemotron-3-Diarization", help="ref: HF repo or checkpoint dir")
    ap.add_argument("--gguf", help="cpp: GGUF path")
    ap.add_argument("--cli", default="build/bin/transcribe-cli", help="cpp: transcribe-cli path")
    ap.add_argument("--backend", default="auto", help="cpp: compute backend")
    ap.add_argument("--threads", type=int, default=0, help="cpp: CPU threads")
    ap.add_argument("--stream-chunk-ms", type=int, default=0, help="cpp: feed the stream API in pieces of N ms")
    ap.add_argument("--rescore-only", action="store_true", help="Rebuild RTTMs from saved probs")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    args.probs_dir.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore_only:
        results = ((row, np.load(args.probs_dir / f"{row['id']}.npy"), 0.0, 0.0) for row in rows)
    elif args.side == "ref":
        results = run_ref(rows, args)
    else:
        if not args.gguf:
            raise SystemExit("error: cpp needs --gguf")
        results = run_cpp(rows, args)

    total_audio = total_time = 0.0
    for row, probs, elapsed, audio_s in results:
        if not args.rescore_only:
            np.save(args.probs_dir / f"{row['id']}.npy", probs.astype(np.float32))
        rttm = probs_to_rttm(probs, row["id"], args.threshold, args.min_dur)
        (args.pred_dir / f"{row['id']}.rttm").write_text(rttm)
        total_audio += audio_s
        total_time += elapsed
        if elapsed:
            print(f"{row['id']}: {audio_s / 60:.1f} min in {elapsed:.1f}s ({audio_s / elapsed:.0f}x)", flush=True)
    if total_time:
        print(f"total: {total_audio / 3600:.2f} h in {total_time / 60:.1f} min ({total_audio / total_time:.0f}x realtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
