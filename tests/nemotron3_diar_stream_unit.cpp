// nemotron3_diar_stream_unit.cpp - Nemotron-3 Diarization push-audio live
// diarization through the Sortformer live stream extension (SFLV, STREAM
// slot).
//
// Covers, against a real GGUF (env-gated, RC 77 skip):
//
//   1. The model accepts SFLV on the STREAM slot only, SFST on RUN only.
//   2. Pre-clear rejection at begin: DEFAULT / VERY_HIGH / HIGH_LATENCY, an
//      out-of-range preset and the RUN-slot kind fail with INVALID_ARG and
//      keep the previous rows.
//   3. Streamed rows after finalize equal transcribe_run at the same preset
//      for feeds of 1 sample to 1 s, for every accepted preset and for the
//      cache-compressing `small` validation geometry.
//   4. Mid-stream: rows only ever report final output. Closed rows
//      (t1_ms < audio_committed_ms) are final rows; open rows
//      (t1_ms == audio_committed_ms) are the start of a final row. The
//      revision advances whenever the rows change.
//   5. No extension runs LOW_LATENCY; reset mid-stream and a stream too short
//      for one mel frame finish cleanly.
//
// Gated by TRANSCRIBE_NEMOTRON3_DIAR_GGUF.

#include "transcribe.h"
#include "transcribe/sortformer.h"
#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<transcribe_speaker_segment> read_segments(const transcribe_session * session) {
    std::vector<transcribe_speaker_segment> rows;
    const int                               n = transcribe_n_speaker_segments(session);
    for (int i = 0; i < n; ++i) {
        transcribe_speaker_segment row;
        transcribe_speaker_segment_init(&row);
        if (transcribe_get_speaker_segment(session, i, &row) == TRANSCRIBE_OK) {
            rows.push_back(row);
        }
    }
    return rows;
}

bool same_row(const transcribe_speaker_segment & a, const transcribe_speaker_segment & b) {
    return a.t0_ms == b.t0_ms && a.t1_ms == b.t1_ms && a.speaker_id == b.speaker_id;
}

bool same_segments(const std::vector<transcribe_speaker_segment> & a,
                   const std::vector<transcribe_speaker_segment> & b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), same_row);
}

std::vector<transcribe_speaker_segment> run_batch(transcribe_session *         session,
                                                  const std::vector<float> &   pcm,
                                                  transcribe_sortformer_preset preset) {
    transcribe_sortformer_stream_ext ext;
    transcribe_sortformer_stream_ext_init(&ext);
    ext.preset = preset;
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.family = &ext.ext;
    CHECK(transcribe_run(session, pcm.data(), static_cast<int>(pcm.size()), &rp) == TRANSCRIBE_OK);
    return read_segments(session);
}

transcribe_status begin(transcribe_session * session, const transcribe_ext * family) {
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.family = family;
    return transcribe_stream_begin(session, nullptr, &sp);
}

transcribe_status begin_preset(transcribe_session * session, transcribe_sortformer_preset preset) {
    transcribe_sortformer_live_ext ext;
    transcribe_sortformer_live_ext_init(&ext);
    ext.preset = preset;
    return begin(session, &ext.ext);
}

// Feeds pcm in `piece`-sample pieces, checking every mid-stream snapshot
// against the batch rows `want`, then finalizes.
std::vector<transcribe_speaker_segment> stream_pieces(transcribe_session *                            session,
                                                      const std::vector<float> &                      pcm,
                                                      size_t                                          piece,
                                                      const std::vector<transcribe_speaker_segment> & want) {
    int64_t                                 last_committed = 0;
    int                                     last_revision  = transcribe_stream_revision(session);
    std::vector<transcribe_speaker_segment> last_rows;
    bool                                    saw_rows = false;
    for (size_t pos = 0; pos < pcm.size(); pos += piece) {
        const size_t             n = std::min(piece, pcm.size() - pos);
        transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        if (transcribe_stream_feed(session, pcm.data() + pos, static_cast<int>(n), &upd) != TRANSCRIBE_OK) {
            CHECK(false);
            return {};
        }
        CHECK(upd.audio_committed_ms >= last_committed);
        CHECK(upd.audio_committed_ms <= upd.input_received_ms);
        last_committed = upd.audio_committed_ms;

        const std::vector<transcribe_speaker_segment> rows = read_segments(session);
        CHECK(upd.result_changed == !same_segments(rows, last_rows));
        CHECK((upd.revision != last_revision) == upd.result_changed);
        last_revision = upd.revision;
        last_rows     = rows;
        saw_rows      = saw_rows || !rows.empty();
        for (const auto & row : rows) {
            CHECK(row.t1_ms <= upd.audio_committed_ms);
            const bool open  = row.t1_ms == upd.audio_committed_ms;
            const bool found = std::any_of(want.begin(), want.end(), [&](const transcribe_speaker_segment & w) {
                return w.speaker_id == row.speaker_id && w.t0_ms == row.t0_ms &&
                       (open ? w.t1_ms >= row.t1_ms : w.t1_ms == row.t1_ms);
            });
            CHECK(found);
        }
    }
    CHECK(saw_rows);
    transcribe_stream_update fin;
    transcribe_stream_update_init(&fin);
    CHECK(transcribe_stream_finalize(session, &fin) == TRANSCRIBE_OK);
    CHECK(fin.is_final);
    CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_FINISHED);
    return read_segments(session);
}

}  // namespace

int main() {
    const char * env = std::getenv("TRANSCRIBE_NEMOTRON3_DIAR_GGUF");
    if (env == nullptr || env[0] == '\0') {
        std::fprintf(stderr,
                     "nemotron3_diar_stream_unit: TRANSCRIBE_NEMOTRON3_DIAR_GGUF not set; skipping.\n"
                     "Re-run with TRANSCRIBE_NEMOTRON3_DIAR_GGUF=models/nemotron-3-diarization/"
                     "nemotron-3-diarization-F32.gguf\n");
        return 77;
    }
    const std::string gguf = env;
    if (!file_exists(gguf)) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: file not found: %s\n", gguf.c_str());
        return 77;
    }
    const std::string  wav_path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/sortformer-2spk-mix.wav";
    std::vector<float> pcm;
    std::string        wav_err;
    if (!transcribe_cli::load_wav_mono_16k(wav_path, pcm, wav_err)) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: wav load: %s\n", wav_err.c_str());
        return 77;
    }
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = TRANSCRIBE_BACKEND_CPU;
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(gguf.c_str(), &mp, &model) != TRANSCRIBE_OK || model == nullptr) {
        std::fprintf(stderr, "FAIL: model load\n");
        return EXIT_FAILURE;
    }

    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE));
    CHECK(transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));
    CHECK(!transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM, TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM));

    transcribe_sortformer_live_ext init;
    transcribe_sortformer_live_ext_init(&init);
    CHECK(init.ext.kind == TRANSCRIBE_EXT_KIND_SORTFORMER_LIVE);
    CHECK(init.ext.size == sizeof(init));
    CHECK(init.preset == TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY);

    struct transcribe_session * session = nullptr;
    if (transcribe_session_init(model, nullptr, &session) != TRANSCRIBE_OK || session == nullptr) {
        std::fprintf(stderr, "FAIL: session create\n");
        transcribe_model_free(model);
        return EXIT_FAILURE;
    }

    const std::vector<transcribe_speaker_segment> low =
        run_batch(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY);
    CHECK(!low.empty());

    // Pre-clear rejection keeps the batch rows.
    for (transcribe_sortformer_preset preset :
         { TRANSCRIBE_SORTFORMER_PRESET_DEFAULT, TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY,
           TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY, static_cast<transcribe_sortformer_preset>(99) }) {
        CHECK(begin_preset(session, preset) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE);
        CHECK(same_segments(read_segments(session), low));
    }
    transcribe_sortformer_stream_ext run_ext;
    transcribe_sortformer_stream_ext_init(&run_ext);
    CHECK(begin(session, &run_ext.ext) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(same_segments(read_segments(session), low));

    // Stream == batch for tiny to 1 s pieces (odd sizes straddle frames).
    for (size_t piece : { size_t{ 1 }, size_t{ 160 }, size_t{ 2401 }, size_t{ 16000 } }) {
        CHECK(begin_preset(session, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY) == TRANSCRIBE_OK);
        CHECK(same_segments(stream_pieces(session, pcm, piece, low), low));
    }
    for (transcribe_sortformer_preset preset :
         { TRANSCRIBE_SORTFORMER_PRESET_VERY_LOW_LATENCY, TRANSCRIBE_SORTFORMER_PRESET_ULTRA_LOW_LATENCY }) {
        const std::vector<transcribe_speaker_segment> want = run_batch(session, pcm, preset);
        CHECK(begin_preset(session, preset) == TRANSCRIBE_OK);
        CHECK(same_segments(stream_pieces(session, pcm, 480, want), want));
    }
    ::setenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET", "small", 1);
    {
        const std::vector<transcribe_speaker_segment> want =
            run_batch(session, pcm, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);
        CHECK(begin_preset(session, TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY) == TRANSCRIBE_OK);
        CHECK(same_segments(stream_pieces(session, pcm, 777, want), want));
    }
    ::unsetenv("TRANSCRIBE_NEMOTRON3_DIAR_PRESET");

    // No extension: LOW_LATENCY.
    CHECK(begin(session, nullptr) == TRANSCRIBE_OK);
    CHECK(same_segments(stream_pieces(session, pcm, 16000, low), low));

    // Reset mid-stream, then a fresh stream.
    CHECK(begin(session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_feed(session, pcm.data(), 48000, nullptr) == TRANSCRIBE_OK);
    transcribe_stream_reset(session);
    CHECK(transcribe_stream_get_state(session) == TRANSCRIBE_STREAM_IDLE);
    CHECK(transcribe_n_speaker_segments(session) == 0);
    CHECK(begin(session, nullptr) == TRANSCRIBE_OK);
    CHECK(same_segments(stream_pieces(session, pcm, 3200, low), low));

    // Shorter than one mel hop: finishes with no rows.
    CHECK(begin(session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_feed(session, pcm.data(), 100, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_stream_finalize(session, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_n_speaker_segments(session) == 0);

    transcribe_session_free(session);
    transcribe_model_free(model);

    if (g_failures != 0) {
        std::fprintf(stderr, "nemotron3_diar_stream_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("nemotron3_diar_stream_unit: OK\n");
    return 0;
}
