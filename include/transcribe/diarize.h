/*
 * include/transcribe/diarize.h - DIARIZE role: who spoke when.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. For models whose
 * product is speaker turns rather than text (transcribe_model_roles() has
 * TRANSCRIBE_ROLE_DIARIZE). ASR models that attribute speakers inside a
 * transcript keep using transcribe_run_params::diarize instead.
 *
 * Open a transcribe_diarize_session on a loaded model, run, then read the
 * speaker segments. Threading and lifetime rules: docs/roles.md.
 */

#ifndef TRANSCRIBE_DIARIZE_H
#define TRANSCRIBE_DIARIZE_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

struct transcribe_diarize_session;

/* Static facts about a diarization model. */
struct transcribe_diarize_info {
    uint64_t struct_size;
    int32_t  sample_rate;  /* input PCM rate (16000) */
    int32_t  max_speakers; /* speaker_id is in [1, max_speakers] */
};

struct transcribe_diarize_session_params {
    uint64_t struct_size;
    int32_t  n_threads; /* 0 = library default */
};

struct transcribe_diarize_params {
    uint64_t                      struct_size;
    /* Typed family extension on TRANSCRIBE_EXT_SLOT_DIARIZE_RUN, or NULL. */
    const struct transcribe_ext * family;
};

TRANSCRIBE_API void transcribe_diarize_info_init(struct transcribe_diarize_info * out);
TRANSCRIBE_API void transcribe_diarize_session_params_init(struct transcribe_diarize_session_params * params);
TRANSCRIBE_API void transcribe_diarize_params_init(struct transcribe_diarize_params * params);

/* UNSUPPORTED_ROLE when the model does not serve TRANSCRIBE_ROLE_DIARIZE. */
TRANSCRIBE_API transcribe_status transcribe_diarize_get_info(const struct transcribe_model *  model,
                                                             struct transcribe_diarize_info * out);

/* params may be NULL for defaults. UNSUPPORTED_ROLE when the model does not
 * serve TRANSCRIBE_ROLE_DIARIZE. On failure *out_session is NULL. */
TRANSCRIBE_API transcribe_status
transcribe_diarize_session_init(struct transcribe_model *                        model,
                                const struct transcribe_diarize_session_params * params,
                                struct transcribe_diarize_session **             out_session);

/* NULL is a no-op. */
TRANSCRIBE_API void transcribe_diarize_session_free(struct transcribe_diarize_session * session);

/* Polled during a run; returning true stops it with TRANSCRIBE_ERR_ABORTED. */
TRANSCRIBE_API void transcribe_diarize_set_abort_callback(struct transcribe_diarize_session * session,
                                                          transcribe_abort_callback           cb,
                                                          void *                              user_data);

/*
 * Diarize one recording: 16 kHz mono float32 PCM, every sample finite.
 * params may be NULL for defaults. Replaces the previous result. Malformed
 * input (NULL pointers, n_samples <= 0, NaN / Inf, a bad extension) returns
 * an error before the previous result is touched.
 */
TRANSCRIBE_API transcribe_status transcribe_diarize_run(struct transcribe_diarize_session *      session,
                                                        const float *                            pcm,
                                                        int                                      n_samples,
                                                        const struct transcribe_diarize_params * params);

/*
 * Speaker segments of the last run, grouped by speaker and time-ordered
 * within a speaker; segments of different speakers may overlap. 0 before any
 * run, and 0 after a run that passed input validation but then failed
 * (aborted, backend error). An out-of-range index returns OK with a zeroed row.
 */
TRANSCRIBE_API int               transcribe_diarize_n_segments(const struct transcribe_diarize_session * session);
TRANSCRIBE_API transcribe_status transcribe_diarize_get_segment(const struct transcribe_diarize_session * session,
                                                                int                                       i,
                                                                struct transcribe_speaker_segment *       out);

/*
 * Push-audio live diarization, for models whose family accepts a kind on
 * TRANSCRIBE_EXT_SLOT_DIARIZE_STREAM (Nemotron-3 Diarization). Feed 16 kHz
 * mono float32 pieces of any size; the rows are readable through
 * transcribe_diarize_n_segments / transcribe_diarize_get_segment after every
 * feed, with the open / closed rule documented by the family extension
 * (include/transcribe/sortformer.h). Updates use the ASR stream's
 * transcribe_stream_update: audio_committed_ms is the processed frontier,
 * input_received_ms the audio fed so far, revision / result_changed advance
 * when the rows change, is_final is set by finalize.
 *
 * begin replaces the previous result once its params validate (a stream
 * already ACTIVE, a bad struct or an unaccepted extension is an error
 * before anything is touched); NOT_IMPLEMENTED when the model has no live
 * path. feed / finalize need an ACTIVE stream; a failure inside them ends
 * the stream (later feeds return INVALID_ARG) and leaves the rows produced
 * so far. A feed with NaN / Inf is rejected and the stream stays ACTIVE.
 * reset ends any stream and clears the rows. A transcribe_diarize_run
 * during an ACTIVE stream returns INVALID_ARG.
 */
struct transcribe_diarize_stream_params {
    uint64_t                      struct_size;
    /* Typed family extension on TRANSCRIBE_EXT_SLOT_DIARIZE_STREAM, or NULL
     * for the family's live default. */
    const struct transcribe_ext * family;
};

TRANSCRIBE_API void transcribe_diarize_stream_params_init(struct transcribe_diarize_stream_params * params);

/* params may be NULL for defaults. */
TRANSCRIBE_API transcribe_status
transcribe_diarize_stream_begin(struct transcribe_diarize_session *             session,
                                const struct transcribe_diarize_stream_params * params);

/* update may be NULL. n_samples must be > 0. */
TRANSCRIBE_API transcribe_status transcribe_diarize_stream_feed(struct transcribe_diarize_session * session,
                                                                const float *                       pcm,
                                                                int                                 n_samples,
                                                                struct transcribe_stream_update *   update);

/* Flushes the buffered tail; every row is final afterwards. update may be NULL. */
TRANSCRIBE_API transcribe_status transcribe_diarize_stream_finalize(struct transcribe_diarize_session * session,
                                                                    struct transcribe_stream_update *   update);

/* NULL is a no-op. */
TRANSCRIBE_API void transcribe_diarize_stream_reset(struct transcribe_diarize_session * session);

/* IDLE before any stream and after reset; ACTIVE between begin and finalize;
 * FINISHED after finalize; FAILED after a feed / finalize error. IDLE for a
 * NULL session. */
TRANSCRIBE_API enum transcribe_stream_state transcribe_diarize_stream_get_state(
    const struct transcribe_diarize_session * session);

/* load_ms plus the last run's stage times. */
TRANSCRIBE_API transcribe_status transcribe_diarize_get_timings(const struct transcribe_diarize_session * session,
                                                                struct transcribe_timings *               out);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_DIARIZE_H */
