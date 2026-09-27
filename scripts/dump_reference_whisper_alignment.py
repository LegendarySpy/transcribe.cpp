# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
dump_reference_whisper_alignment.py - golden values for the Whisper word
timestamp host math (src/arch/whisper/alignment.cpp).

The functions below are direct numpy ports of openai/whisper timing.py
(median_filter, dtw_cpu, backtrace, find_alignment's post-processing,
merge_punctuations, add_word_timestamps' duration and segment rules).
numpy "reflect" padding equals torch "reflect". Word times are not rounded
to 0.01 s (the C++ port keeps full precision, a documented deviation).

    uv run scripts/dump_reference_whisper_alignment.py \
      > tests/fixtures/whisper_alignment_ref.inc
"""

import numpy as np

TOKENS_PER_SECOND = 50
PREPENDED = "\"'“¿([{-"
APPENDED = "\"'.。,，!！?？:：”)]}、"
SENTENCE_END = ".。!！?？"


# ---- timing.py ports -------------------------------------------------------


def median_filter(x: np.ndarray, filter_width: int) -> np.ndarray:
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        return x
    pad = [(0, 0)] * (x.ndim - 1) + [(pad_width, pad_width)]
    xp = np.pad(x, pad, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(xp, filter_width, axis=-1)
    return np.sort(win, axis=-1)[..., pad_width]


def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1
    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")
    result = np.array(result)
    return result[::-1, :].T


def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    return backtrace(trace)


def alignment_matrix(qk: np.ndarray, sot_len: int, qk_scale: float) -> np.ndarray:
    """find_alignment from the stacked QKs to `matrix` (float32, like torch)."""
    w = (qk.astype(np.float32) * np.float32(qk_scale)).astype(np.float32)
    w = w - w.max(axis=-1, keepdims=True)
    w = np.exp(w)
    w = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
    mean = w.mean(axis=-2, keepdims=True)
    std = w.std(axis=-2, keepdims=True)  # unbiased=False
    w = ((w - mean) / std).astype(np.float32)
    w = median_filter(w, 7)
    matrix = w.mean(axis=0)
    return matrix[sot_len:-1].astype(np.float32)


def jump_frames(matrix: np.ndarray):
    text_indices, time_indices = dtw_cpu((-matrix).astype(np.float64))
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    return time_indices[jumps]


def duration_heuristics(words):
    durations = np.array([w["end"] - w["start"] for w in words])
    durations = durations[durations.nonzero()]
    median = np.median(durations) if len(durations) > 0 else 0.0
    median = min(0.7, float(median))
    max_duration = median * 2
    if len(durations) > 0:
        for i in range(1, len(words)):
            if words[i]["end"] - words[i]["start"] > max_duration:
                if words[i]["word"] in SENTENCE_END:
                    words[i]["end"] = words[i]["start"] + max_duration
                elif words[i - 1]["word"] in SENTENCE_END:
                    words[i]["start"] = words[i]["end"] - max_duration
    return median, max_duration


def merge_punctuations(alignment):
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in PREPENDED:
            following["word"] = previous["word"] + following["word"]
            previous["word"] = ""
        else:
            j = i
        i -= 1
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in APPENDED:
            previous["word"] = previous["word"] + following["word"]
            following["word"] = ""
        else:
            i = j
        j += 1


def segment_rules(segment, words, median_duration, max_duration, last_speech_timestamp):
    if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
        words[0]["end"] - words[0]["start"] > max_duration
        or (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)
    ):
        if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
            boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
            words[0]["end"] = words[1]["start"] = boundary
        words[0]["start"] = max(0, words[0]["end"] - max_duration)
    if segment["start"] < words[0]["end"] and segment["start"] - 0.5 > words[0]["start"]:
        words[0]["start"] = max(0, min(words[0]["end"] - median_duration, segment["start"]))
    else:
        segment["start"] = words[0]["start"]
    if segment["end"] > words[-1]["start"] and segment["end"] + 0.5 < words[-1]["end"]:
        words[-1]["end"] = max(words[-1]["start"] + median_duration, segment["end"])
    else:
        segment["end"] = words[-1]["end"]
    return segment["end"]


# ---- C++ emission ----------------------------------------------------------


def f32(v) -> str:
    t = f"{float(np.float32(v)):.9g}"
    if not any(c in t for c in ".en"):
        t += ".0"
    return t + "f"


def f64(v) -> str:
    return f"{float(v):.17g}"


def floats(a, fmt=f32) -> str:
    return "{ " + ", ".join(fmt(v) for v in np.asarray(a).ravel()) + " }"


def ints(a) -> str:
    return "{ " + ", ".join(str(int(v)) for v in np.asarray(a).ravel()) + " }"


def cstr(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        if 0x20 <= b < 0x7F and chr(b) not in "\"\\":
            out.append(chr(b))
        else:
            out.append(f'\\x{b:02x}" "')
    return '"' + "".join(out) + '"'


def main():
    rng = np.random.default_rng(20260926)
    print("// Auto-generated by scripts/dump_reference_whisper_alignment.py.")
    print("// Regenerate with:")
    print("//   uv run scripts/dump_reference_whisper_alignment.py \\")
    print("//     > tests/fixtures/whisper_alignment_ref.inc")
    print()

    print("const MedfiltCase k_medfilt_cases[] = {")
    for T in [1, 2, 3, 4, 5, 6, 7, 8, 20]:
        x = rng.standard_normal(T).astype(np.float32)
        y = median_filter(x[None, None, :], 7)[0, 0]
        print(f"    {{ {T}, {floats(x)}, {floats(y)} }},")
    print("};")
    print()

    print("const DtwCase k_dtw_cases[] = {")
    shapes = [(3, 5), (5, 3), (1, 4), (4, 1), (1, 1), (6, 9), (8, 8)]
    for N, M in shapes:
        x = rng.standard_normal((N, M)).astype(np.float32)
        ti, mi = dtw_cpu(x.astype(np.float64))
        print(f"    {{ {N}, {M}, {floats(x)}, {ints(ti)}, {ints(mi)}, {len(ti)} }},")
    # Ties: every cost equal, so the tie-break order decides the path.
    for N, M in [(3, 3), (2, 4)]:
        x = np.zeros((N, M), dtype=np.float32)
        ti, mi = dtw_cpu(x.astype(np.float64))
        print(f"    {{ {N}, {M}, {floats(x)}, {ints(ti)}, {ints(mi)}, {len(ti)} }},")
    print("};")
    print()

    print("const PipelineCase k_pipeline_cases[] = {")
    for H, R, T, sot_len in [(3, 9, 12, 3), (1, 4, 5, 1), (2, 6, 3, 3), (6, 20, 40, 3), (2, 5, 4, 1)]:
        qk = (rng.standard_normal((H, R, T)) * 4.0).astype(np.float32)
        # Give the diagonal a lift so the path is well defined.
        for r in range(R):
            qk[:, r, min(T - 1, r * T // R)] += 6.0
        scale = 0.125
        m = alignment_matrix(qk, sot_len, scale)
        jf = jump_frames(m)
        print(f"    {{ {H}, {R}, {T}, {sot_len}, {f32(scale)}, {floats(qk)}, {floats(m)}, {ints(jf)} }},")
    print("};")
    print()

    # Duration heuristics + punctuation merge on window-relative words.
    word_sets = [
        [(" Hello", 0.0, 0.4), (",", 0.4, 0.5), (" world", 0.5, 2.9), (".", 2.9, 3.0)],
        [(" \"", 0.0, 0.1), (" Quote", 0.1, 0.5), ("\"", 0.5, 0.6), (" end", 0.6, 3.5)],
        [(" (", 1.0, 1.1), (" a", 1.1, 1.3), (")", 1.3, 1.4), (" b", 1.4, 1.6)],
        [(" Stop", 0.0, 0.3), (".", 0.3, 0.3), (" Next", 0.3, 4.0), (" one", 4.0, 4.2)],
        [(" x", 0.0, 0.2), (" y", 0.2, 0.2), (" z", 0.2, 0.6)],
        [(" a", 0.0, 0.0), (" b", 0.0, 0.0)],
        [(" one", 0.0, 0.2), (" two", 0.2, 0.5), (" three", 0.5, 0.6), (" four", 0.6, 1.0)],
        [(" ", 0.0, 0.1), ("-", 0.1, 0.2), (" dash", 0.2, 0.4), ("?", 0.4, 3.0)],
    ]
    print("const WordsCase k_words_cases[] = {")
    for ws in word_sets:
        words = [{"word": w, "start": s, "end": e} for w, s, e in ws]
        median, maxd = duration_heuristics(words)
        merge_punctuations(words)
        src = ", ".join(f"{{ {cstr(w)}, {f64(s)}, {f64(e)} }}" for w, s, e in ws)
        exp = ", ".join(f"{{ {cstr(w['word'])}, {f64(w['start'])}, {f64(w['end'])} }}" for w in words)
        print(f"    {{ {len(ws)}, {{ {src} }}, {{ {exp} }}, {f64(median)}, {f64(maxd)} }},")
    print("};")
    print()

    # Segment rules (absolute seconds).
    seg_sets = [
        # (seg_start, seg_end, words, median, max, last_speech)
        (0.0, 3.0, [(0.0, 2.5), (2.5, 2.8), (2.8, 3.0)], 0.3, 0.6, 0.0),
        (5.0, 7.0, [(4.0, 5.2), (5.2, 6.8)], 0.3, 0.6, 1.0),
        (1.0, 2.0, [(1.1, 1.3), (1.3, 3.0)], 0.3, 0.6, 1.0),
        (2.0, 4.0, [(0.5, 2.6), (2.6, 3.9)], 0.4, 0.8, 0.0),
        (0.0, 1.0, [(0.2, 0.9)], 0.3, 0.6, 0.0),
        (10.0, 12.0, [(10.0, 10.3), (10.3, 10.6)], 0.3, 0.6, 10.0),
    ]
    print("const SegCase k_seg_cases[] = {")
    for ss, se, ws, med, mx, last in seg_sets:
        words = [{"start": s, "end": e} for s, e in ws]
        seg = {"start": ss, "end": se}
        new_last = segment_rules(seg, words, med, mx, last)
        src = ", ".join(f"{{ {f64(s)}, {f64(e)} }}" for s, e in ws)
        exp = ", ".join(f"{{ {f64(w['start'])}, {f64(w['end'])} }}" for w in words)
        print(
            f"    {{ {f64(ss)}, {f64(se)}, {len(ws)}, {{ {src} }}, {f64(med)}, {f64(mx)}, {f64(last)}, "
            f"{{ {exp} }}, {f64(seg['start'])}, {f64(seg['end'])}, {f64(new_last)} }},"
        )
    print("};")


if __name__ == "__main__":
    main()
