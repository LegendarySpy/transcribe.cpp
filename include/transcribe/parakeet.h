/*
 * include/transcribe/parakeet.h - Parakeet-family public extension surface.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming extension structs (cache-aware and chunked-attention
 * variants), the run extension (phrase boosting), and their kind
 * constants and init functions.
 *
 * Acceptance is per-loaded-model-variant: nemotron-speech-streaming-en-0.6b
 * (cache-aware) accepts TRANSCRIBE_EXT_KIND_PARAKEET_STREAM and rejects
 * TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM; parakeet-unified-en-0.6b
 * (chunked_limited_with_rc) does the opposite. Probe via
 * transcribe_model_accepts_ext_kind before pointing
 * transcribe_stream_params::family at one of these structs.
 *
 * FourCC kinds are reserved in docs/extension-kinds.md.
 */

#ifndef TRANSCRIBE_PARAKEET_H
#define TRANSCRIBE_PARAKEET_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'PKST' little-endian = 0x54534B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_STREAM          0x54534B50u
/* 'PKBS' little-endian = 0x53424B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM 0x53424B50u
/* 'PKRN' little-endian = 0x4E524B50 */
#define TRANSCRIBE_EXT_KIND_PARAKEET_RUN             0x4E524B50u

/*
 * Cache-aware streaming knob (nemotron-speech-streaming-en-0.6b).
 *
 *   att_context_right
 *
 *     Right-context (lookahead) selector in encoder frames. The cache-
 *     aware streaming variants are trained on a menu of (left, right)
 *     pairs simultaneously - the user picks one at inference time to
 *     trade latency for accuracy. nemotron's published menu is
 *     right ∈ {13, 6, 1, 0}, corresponding to lookahead of
 *     {1040, 480, 80, 0} ms at the 80ms encoder frame rate.
 *
 *     -1 (default): use the model's default setting (first entry of
 *                   att_context_size_choices = max-accuracy /
 *                   max-latency).
 *     < -1:         caller bug; transcribe_stream_begin returns
 *                   TRANSCRIBE_ERR_INVALID_ARG.
 *     >= 0:         select the corresponding (left, att_context_right)
 *                   entry from the model's training menu. 0 is
 *                   legitimate when 0-frame lookahead is in the menu.
 *                   transcribe_stream_begin returns
 *                   TRANSCRIBE_ERR_INVALID_ARG if the requested right
 *                   is not in the menu.
 *
 *     The published menu and the lookahead (in milliseconds) each
 *     entry corresponds to are documented in the model's family doc
 *     (docs/models/nemotron-speech-streaming-en-0.6b.md); -1 selects
 *     the model's default (max-accuracy / max-latency) entry.
 */
struct transcribe_parakeet_stream_ext {
    struct transcribe_ext ext;
    int32_t               att_context_right;
};

/* Fills ext.size/kind and att_context_right = -1 (model default). */
TRANSCRIBE_API void transcribe_parakeet_stream_ext_init(struct transcribe_parakeet_stream_ext * ext);

/*
 * Chunked-attention (buffered) streaming knob (parakeet-unified-en-0.6b).
 *
 * parakeet-unified-en-0.6b is trained with chunked_limited_with_rc
 * attention over a menu of (left, chunk, right) context tuples
 * expressed in 80ms encoder frames. The user picks the active tuple at
 * stream_begin time; the encoder re-runs over each new
 * [left | chunk | right] PCM window. Each field is in MILLISECONDS;
 * the runtime converts to encoder frames at the model's frame rate.
 *
 * Per-field sentinels (each field independently):
 *
 *   -1     "use the model default for this field." Unified-en-0.6b's
 *          best-accuracy default is L=5600 ms / C=1040 ms / R=1040 ms,
 *          which the published WER numbers correspond to.
 *   < -1   caller bug; transcribe_stream_begin returns
 *          TRANSCRIBE_ERR_INVALID_ARG.
 *   0      a legitimate requested value when 0 frames is in the model's
 *          menu for that field. Not all fields admit 0.
 *   > 0    must be an exact positive multiple of the encoder frame size
 *          (80 ms for every shipped FastConformer streaming variant).
 *          A value that does not divide the frame returns
 *          TRANSCRIBE_ERR_INVALID_ARG; the runtime never silently floors.
 *
 * After per-field resolution the (L, C, R) frame tuple is validated
 * against the model's training menu
 * (stt.parakeet.encoder.att_chunk_{left,chunk,right}_choices); tuples
 * outside the menu return TRANSCRIBE_ERR_INVALID_ARG.
 */
struct transcribe_parakeet_buffered_stream_ext {
    struct transcribe_ext ext;
    int32_t               left_ms;
    int32_t               chunk_ms;
    int32_t               right_ms;
};

/* Fills ext.size/kind and left/chunk/right_ms = -1 (model default). */
TRANSCRIBE_API void transcribe_parakeet_buffered_stream_ext_init(struct transcribe_parakeet_buffered_stream_ext * ext);

/*
 * Phrase boosting (RUN slot; also honored by transcribe_stream_begin
 * through its run_params). Accepted by TDT and RNN-T variants, not CTC.
 *
 *   boost_phrases / n_boost_phrases
 *
 *     UTF-8 phrases to favor, as typed (casing matters: the model emits
 *     cased text, so "Groq" boosts "Groq"). Borrowed for the call; the
 *     library copies what it needs. NULL / 0 disables boosting and the
 *     output is identical to passing no extension. At most 1024
 *     phrases; more, or a NULL entry, is TRANSCRIBE_ERR_INVALID_ARG.
 *     Whitespace is trimmed and collapsed. Phrases of fewer than 3
 *     characters or more than 256 bytes, longer than 24 tokens, or that
 *     do not tokenize to plain vocabulary pieces, are ignored.
 *
 *   boost_score
 *
 *     Boost weight in logits per phrase-trie arc score. Default 3.0;
 *     0 disables. Must be finite and in [0, 10].
 *
 * Greedy decoding keeps the model's blank decisions. A boosted token must
 * have at least 1e-5 unboosted probability, and a swap is kept only when
 * it completes a phrase as whole words and the decode keeps emitting
 * speech for half a second after it; otherwise the model's own tokens
 * stand for that span.
 */
struct transcribe_parakeet_run_ext {
    struct transcribe_ext ext;
    const char * const *  boost_phrases;
    int32_t               n_boost_phrases;
    float                 boost_score;
};

/* Fills ext.size/kind, no phrases, boost_score = 3.0. */
TRANSCRIBE_API void transcribe_parakeet_run_ext_init(struct transcribe_parakeet_run_ext * ext);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TRANSCRIBE_PARAKEET_H */
