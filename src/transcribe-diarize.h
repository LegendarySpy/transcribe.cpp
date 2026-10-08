// transcribe-diarize.h - internal DIARIZE role surface: the session base,
// the per-arch ops table, and the shared probs -> segments step.
//
// Families compute frame probabilities; the role dispatcher
// (transcribe-diarize.cpp) turns them into speaker segments.

#pragma once

#include "transcribe-session-core.h"
#include "transcribe/diarize.h"

#include <vector>

struct transcribe_diarize_session : transcribe::SessionCore {
    std::vector<transcribe::SpeakerSegmentEntry> segments;  // last successful run's rows (or the live rows)

    // Push-audio stream lifecycle (transcribe_diarize_stream_*), owned by
    // the dispatcher; the family hooks own the rows and the revision.
    enum transcribe_stream_state stream_state    = TRANSCRIBE_STREAM_IDLE;
    int32_t                      stream_revision = 0;
};

namespace transcribe {

// One run's family output: probs is row-major [n_frames, n_speakers].
struct DiarizeProbs {
    std::vector<float> probs;
    int                n_frames   = 0;
    int                n_speakers = 0;
    double             frame_ms   = 0.0;
};

struct DiarizeOps {
    int (*max_speakers)(const transcribe_model * model);
    transcribe_diarize_session * (*new_session)();
    // Pure pre-clear check of params->family (already kind/size-checked);
    // NULL = nothing to check.
    transcribe_status (*run_validate)(const transcribe_diarize_params * params);
    transcribe_status (*run)(transcribe_diarize_session *      session,
                             const float *                     pcm,
                             int                               n_samples,
                             const transcribe_diarize_params * params,
                             DiarizeProbs &                    out);

    // Optional push-audio streaming (NULL = no live path). validate is a
    // pure pre-clear check of params->family (already kind/size-checked);
    // begin / feed / finalize are a required triple that publish rows into
    // session->segments and fill `update` (nullable) with the cursors,
    // bumping session->stream_revision when the rows change; reset drops
    // per-stream state. The dispatcher owns stream_state.
    transcribe_status (*stream_validate)(const transcribe_diarize_session *       session,
                                         const transcribe_diarize_stream_params * params);
    transcribe_status (*stream_begin)(transcribe_diarize_session *             session,
                                      const transcribe_diarize_stream_params * params);
    transcribe_status (*stream_feed)(transcribe_diarize_session * session,
                                     const float *                pcm,
                                     int                          n_samples,
                                     transcribe_stream_update *   update);
    transcribe_status (*stream_finalize)(transcribe_diarize_session * session, transcribe_stream_update * update);
    void (*stream_reset)(transcribe_diarize_session * session);
};

// Append one segment per contiguous run of probs > 0.5 per speaker,
// speaker-major and time-ordered (speaker_id 1-based, p = NaN).
void probs_to_segments(const float *                      probs,
                       int                                n_frames,
                       int                                n_speakers,
                       double                             frame_ms,
                       std::vector<SpeakerSegmentEntry> & out);

}  // namespace transcribe
