/*
 * include/transcribe/sortformer.h - Sortformer-family public extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming-operating-point run extension, the push-audio stream
 * extension, their kind constants, and their init functions.
 *
 * Sortformer (diar_streaming_sortformer_4spk-v2.1) is a diarization-only model: a run
 * produces no text; the product is the who-spoke-when rows read back via
 * transcribe_n_speaker_segments / transcribe_get_speaker_segment
 * (TRANSCRIBE_FEATURE_DIARIZATION). The compute core is streaming
 * (AOSC speaker cache + FIFO); the batch transcribe_run over a whole
 * recording takes the RUN-slot extension (SFST). Push-audio live
 * diarization (transcribe_stream_begin / feed / finalize) takes the
 * separate STREAM-slot extension (SFLV) with the same preset enum; only
 * Nemotron-3 Diarization implements it.
 *
 * Probe via transcribe_model_accepts_ext_kind(model,
 * TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM)
 * before pointing transcribe_run_params::family at the run struct, and
 * (TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE)
 * before pointing transcribe_stream_params::family at the live struct.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_SORTFORMER_H
#define TRANSCRIBE_SORTFORMER_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'SFST' little-endian = 0x54534653 */
#define TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM 0x54534653u

/*
 * Streaming operating point (latency / accuracy trade-off).
 *
 * The model processes audio in fixed chunks and carries speaker identity
 * across chunks in a bounded cache; the operating point sets the chunk
 * geometry. Each named preset is a jointly-tuned bundle published by the
 * upstream model (chunk length, lookahead, FIFO and speaker-cache
 * geometry) - the menu is discrete, not a continuous latency dial, and
 * only these bundles are accuracy-validated (AMI DER, see
 * docs/porting/families/sortformer.md).
 *
 *   DEFAULT             The GGUF-shipped checkpoint configuration.
 *   VERY_HIGH_LATENCY   ~30.4 s algorithmic lookahead (chunk 340 + rc 40
 *                       frames @ 80 ms). Highest accuracy; the published
 *                       operating point for offline file processing.
 *   HIGH_LATENCY        ~10.0 s lookahead (chunk 124 + rc 1).
 *   LOW_LATENCY         ~1.04 s lookahead (chunk 6 + rc 7). The
 *                       real-time operating point. Note: much higher
 *                       compute per audio second than the larger chunks
 *                       (many small windows); see the family doc for
 *                       measured throughput.
 *   VERY_LOW_LATENCY    ~0.64 s lookahead (Nemotron-3 Diarization only).
 *   ULTRA_LOW_LATENCY   ~0.32 s lookahead (Nemotron-3 Diarization only).
 *
 * Nemotron-3 Diarization (arch nemotron3_diar) shares this extension. Its
 * presets are NVIDIA's published v3 operating points: VERY_HIGH_LATENCY is
 * 30.4 s (chunk 340 + rc 40), LOW_LATENCY 1.04 s (chunk 9 + rc 4); it has no
 * HIGH_LATENCY point. A preset a model has no validated geometry for is
 * rejected with TRANSCRIBE_ERR_INVALID_ARG.
 *
 * Values outside the enum range are rejected by transcribe_run with
 * TRANSCRIBE_ERR_INVALID_ARG before the previous result is cleared.
 */
typedef enum {
    TRANSCRIBE_SORTFORMER_PRESET_DEFAULT           = 0,
    TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY = 1,
    TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY      = 2,
    TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY       = 3,
    TRANSCRIBE_SORTFORMER_PRESET_VERY_LOW_LATENCY  = 4,
    TRANSCRIBE_SORTFORMER_PRESET_ULTRA_LOW_LATENCY = 5,
} transcribe_sortformer_preset;

struct transcribe_sortformer_stream_ext {
    struct transcribe_ext        ext;
    transcribe_sortformer_preset preset;
};

/* Fills ext.size/kind and preset = DEFAULT (GGUF-shipped cfg). */
TRANSCRIBE_API void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * ext);

/* 'SFLV' little-endian = 0x564C4653 */
#define TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE 0x564C4653u

/*
 * Push-audio live diarization (STREAM slot, Nemotron-3 Diarization only).
 *
 * Feed 16 kHz mono f32 pieces of any size; a chunk runs as soon as the
 * chunk plus its lookahead has arrived. Accepted presets are
 * the real-time ones: LOW_LATENCY (the init default), VERY_LOW_LATENCY
 * and ULTRA_LOW_LATENCY. DEFAULT, VERY_HIGH_LATENCY and HIGH_LATENCY are
 * rejected with TRANSCRIBE_ERR_INVALID_ARG before the previous result is
 * cleared; use transcribe_run with the SFST extension for those. A stream
 * begun without an extension runs LOW_LATENCY.
 *
 * After transcribe_stream_finalize the rows equal a transcribe_run at the
 * same preset. During the stream they are readable after every feed:
 *
 *   - transcribe_stream_update::audio_committed_ms is the processed
 *     frontier: the model's output is final up to there and nothing past
 *     it is reported yet.
 *   - A row with t1_ms < audio_committed_ms is closed and final.
 *   - A row with t1_ms == audio_committed_ms is open: the speaker is still
 *     talking at the frontier, and a later feed extends t1_ms (or closes
 *     the row there). Open rows exist only mid-stream; after finalize
 *     every row is final.
 *
 * The rule needs the default commit policy: ON_FINALIZE reports
 * audio_committed_ms = 0 during feeds. Rows are ordered by speaker, then
 * time, as for transcribe_run; result_changed / the stream revision
 * advance whenever the rows change.
 */
struct transcribe_sortformer_live_ext {
    struct transcribe_ext        ext;
    transcribe_sortformer_preset preset;
};

/* Fills ext.size/kind and preset = LOW_LATENCY. */
TRANSCRIBE_API void transcribe_sortformer_live_ext_init(struct transcribe_sortformer_live_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_SORTFORMER_H */
