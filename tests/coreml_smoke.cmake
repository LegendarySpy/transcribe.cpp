# Real GGUF + its derived Core ML encoder for one family, supplied through
# TRANSCRIBE_<FAMILY>_COREML_GGUF and TRANSCRIBE_<FAMILY>_COREML_MODEL.
set(gguf_var "TRANSCRIBE_${FAMILY}_COREML_GGUF")
set(model_var "TRANSCRIBE_${FAMILY}_COREML_MODEL")
set(model "$ENV{${gguf_var}}")
set(encoder "$ENV{${model_var}}")
if(model STREQUAL "" OR encoder STREQUAL "")
    message("SKIP: set ${gguf_var} and ${model_var}")
    return()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=${model_var}
        "${CLI}" -m "${model}" --backend cpu --threads 2 "${AUDIO}"
    RESULT_VARIABLE status OUTPUT_VARIABLE cpu ERROR_VARIABLE diagnostics)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "CPU baseline failed: ${diagnostics}")
endif()

execute_process(
    COMMAND "${CLI}" -m "${model}" --backend cpu --threads 2 --repeat 2 "${AUDIO}"
    RESULT_VARIABLE status OUTPUT_VARIABLE coreml ERROR_VARIABLE diagnostics)
if(NOT status EQUAL 0 OR NOT diagnostics MATCHES "Core ML encoder loaded")
    message(FATAL_ERROR "Core ML session reuse failed: ${diagnostics}")
endif()
string(REGEX MATCH "text: [^\n]+" cpu_text "${cpu}")
string(REGEX MATCH "text: [^\n]+" coreml_text "${coreml}")
if(cpu_text STREQUAL "" OR NOT cpu_text STREQUAL coreml_text)
    message(FATAL_ERROR "Transcript mismatch: CPU=${cpu_text}; Core ML=${coreml_text}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "${model_var}=${encoder}/missing.mlmodelc"
        "${CLI}" -m "${model}" --backend cpu --threads 2 "${AUDIO}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE diagnostics)
if(status EQUAL 0 OR NOT diagnostics MATCHES "Core ML encoder: load failed")
    message(FATAL_ERROR "Invalid Core ML path did not fail explicitly: ${output} ${diagnostics}")
endif()
message("Core ML transcript parity, session reuse, and invalid-path checks passed")
