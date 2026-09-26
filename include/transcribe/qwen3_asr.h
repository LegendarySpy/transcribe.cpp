#ifndef TRANSCRIBE_QWEN3_ASR_H
#define TRANSCRIBE_QWEN3_ASR_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'QWRN' little-endian. Accepted by Qwen3-ASR in the RUN slot. */
#define TRANSCRIBE_EXT_KIND_QWEN3_ASR_RUN 0x4E525751u

struct transcribe_qwen3_asr_run_ext {
    struct transcribe_ext ext;
    /* UTF-8 vocabulary/background context for the system turn, not cleanup
     * instructions. NULL/empty preserves the default prompt. Borrowed for
     * the call. At most 4096 bytes and 1024 tokens; excess is INVALID_ARG. */
    const char *          context;
};

TRANSCRIBE_API void transcribe_qwen3_asr_run_ext_init(struct transcribe_qwen3_asr_run_ext * ext);

#ifdef __cplusplus
}
#endif
#endif
