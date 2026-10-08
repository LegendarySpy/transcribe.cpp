// arch/whisper/capabilities.cpp - Whisper ASR capability defaults.

#include "transcribe-log.h"
#include "whisper.h"

namespace transcribe::whisper {

void apply_family_invariants(transcribe_model & model) {
    transcribe_capabilities & caps = model.caps;

    caps.native_sample_rate = 16000;

    // Multilingual defaults; read_capability_kv() in the load path overrides
    // these from GGUF (.en variants carry lang_detect/translate=false).
    caps.supports_language_detect = true;
    caps.supports_translate       = true;
    // supports_streaming left at its zero-init default (false).

    // Segment timestamps from the timestamp tokens, word timestamps from a
    // cross-attention alignment pass. AUTO still resolves to SEGMENT; words
    // only on an explicit WORD request.
    caps.max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_WORD;

    // Run knobs are reached through transcribe_whisper_run_ext. No PNC/ITN
    // runtime toggle — whisper emits whatever its training distribution does.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_INITIAL_PROMPT, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_LONG_FORM, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CANCELLATION, true);
    // Generic vocabulary (`Glossary: {terms}`) and context prompt, both in the
    // <|startofprev|> slot (prompting A/B, notes/prompting-ab-results.md).
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_VOCABULARY, true);
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_CONTEXT_PROMPT, true);
    // Transcript prefix after the SOT sequence (openai DecodingOptions.prefix),
    // first window only.
    transcribe::set_feature(&model, TRANSCRIBE_FEATURE_TRANSCRIPT_PREFIX, true);
}

void resolve_alignment_heads(WhisperModel & m) {
    const WhisperHParams & hp = m.hparams;
    AlignGeometry          g;
    g.enc_n_layers = hp.enc_n_layers;
    g.dec_n_layers = hp.dec_n_layers;
    g.dec_n_heads  = hp.dec_n_heads;
    g.n_mels       = hp.enc_num_mel_bins;
    g.multilingual = hp.dec_vocab_size >= 51865;
    std::string source;
    m.align_heads = resolve_alignment_heads(g, m.variant, hp.alignment_heads, source);
    if (!hp.alignment_heads.empty() && source != "gguf") {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "whisper: ignoring invalid stt.whisper.alignment_heads");
    }
    if (!m.align_heads.empty()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "whisper: alignment heads = %s (%zu heads, layers %d..%d)", source.c_str(),
                m.align_heads.size(), m.align_heads.front().layer, m.align_heads.back().layer);
    }
}

}  // namespace transcribe::whisper
