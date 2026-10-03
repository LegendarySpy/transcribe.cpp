#include "transcribe-coreml.h"

#include "transcribe-log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace transcribe {

// One fixed-shape program of the companion: its default function or a
// smaller extra one.
struct CoreMLProgram {
    MLModel *      model        = nil;
    MLMultiArray * input        = nil;
    int            capacity     = 0;
    int            capacity_out = 0;
};

struct CoreMLEncoder {
    std::vector<CoreMLProgram> programs;  // ascending capacity, the default function last
    MLMultiArray *             length       = nil;
    int                        n_mels       = 0;
    int                        capacity     = 0;
    int                        d_model      = 0;
    int                        subsampling  = 0;
    int                        capacity_out = 0;
};

static void log_error(const char * operation, NSError * error) {
    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "Core ML encoder: %s: %s", operation,
            error ? error.localizedDescription.UTF8String : "invalid model tensor contract");
}

// Checks a loaded program's tensor contract and allocates its input. capacity
// > 0 must match; capacity_out 0 derives it from the capacity, < 0 reads it
// from the output shape.
static bool make_program(MLModel *       model,
                         int             n_mels,
                         int             capacity,
                         int             d_model,
                         int             subsampling,
                         int             capacity_out,
                         CoreMLProgram & program) {
    MLModelDescription *     desc   = model.modelDescription;
    MLMultiArrayConstraint * input  = desc.inputDescriptionsByName[@"logmel_data"].multiArrayConstraint;
    MLMultiArrayConstraint * output = desc.outputDescriptionsByName[@"output"].multiArrayConstraint;
    if (!input || input.shape.count != 3 || input.shape[0].intValue != 1 || input.shape[1].intValue != n_mels ||
        input.shape[2].longLongValue <= 0 ||
        input.shape[2].longLongValue > std::numeric_limits<int>::max() - subsampling ||
        (capacity > 0 && input.shape[2].intValue != capacity)) {
        log_error("input shape mismatch", nil);
        return false;
    }
    capacity = input.shape[2].intValue;
    if (capacity_out == 0) {
        capacity_out = (capacity + subsampling - 1) / subsampling;
    } else if (capacity_out < 0 && output && output.shape.count == 3) {
        capacity_out = output.shape[1].intValue;
    }
    if (!output || capacity_out <= 0 || ![output.shape isEqualToArray:@[ @1, @(capacity_out), @(d_model) ]] ||
        input.dataType != MLMultiArrayDataTypeFloat32 || output.dataType != MLMultiArrayDataTypeFloat32) {
        log_error("output shape or dtype mismatch", nil);
        return false;
    }
    NSError * error = nil;
    program.input = [[MLMultiArray alloc] initWithShape:input.shape dataType:MLMultiArrayDataTypeFloat32 error:&error];
    if (!program.input) {
        log_error("input allocation failed", error);
        return false;
    }
    program.model        = model;
    program.capacity     = capacity;
    program.capacity_out = capacity_out;
    return true;
}

CoreMLEncoder * coreml_encoder_load(const char * path,
                                    const char * variant,
                                    int          n_mels,
                                    int          capacity,
                                    int          d_model,
                                    int          subsampling,
                                    bool         variable_length,
                                    int          capacity_out) {
    @autoreleasepool {
        @try {
            if (@available(macOS 13.0, *)) {
                auto       encoder    = std::make_unique<CoreMLEncoder>();
                NSString * model_path = [NSString stringWithUTF8String:path];
                if (!model_path || n_mels <= 0 || d_model <= 0 || subsampling <= 0) {
                    log_error("invalid configuration", nil);
                    return nullptr;
                }
                MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
                config.computeUnits           = MLComputeUnitsCPUAndNeuralEngine;
                NSError * error               = nil;
                NSURL *   url                 = [NSURL fileURLWithPath:model_path];
                MLModel * model               = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
                if (!model) {
                    log_error("load failed", error);
                    return nullptr;
                }
                MLModelDescription * desc           = model.modelDescription;
                // One variant, or a comma-separated list for encoders shared across checkpoints.
                NSString *           source_variant = desc.metadata[MLModelCreatorDefinedKey][@"transcribe.variant"];
                NSArray *            variants       = [source_variant componentsSeparatedByString:@","];
                if (![variants containsObject:[NSString stringWithUTF8String:variant]]) {
                    log_error("encoder checkpoint does not match GGUF variant", nil);
                    return nullptr;
                }
                CoreMLProgram main_program;
                if (!make_program(model, n_mels, capacity, d_model, subsampling, capacity_out, main_program)) {
                    return nullptr;
                }
                if (variable_length) {
                    MLMultiArrayConstraint * length = desc.inputDescriptionsByName[@"mel_length"].multiArrayConstraint;
                    if (!length || ![length.shape isEqualToArray:@[ @1 ]] ||
                        length.dataType != MLMultiArrayDataTypeInt32) {
                        log_error("missing mel_length input", nil);
                        return nullptr;
                    }
                    encoder->length = [[MLMultiArray alloc] initWithShape:@[ @1 ]
                                                                 dataType:MLMultiArrayDataTypeInt32
                                                                    error:&error];
                    if (!encoder->length) {
                        log_error("length allocation failed", error);
                        return nullptr;
                    }
                }
                // A variable-length companion may add smaller fixed-shape functions
                // (a macOS 15 multifunction model, listed in transcribe.functions);
                // a short input runs on the smallest one that holds it.
                NSString * functions = desc.metadata[MLModelCreatorDefinedKey][@"transcribe.functions"];
                if (variable_length && functions.length > 0) {
                    if (@available(macOS 15.0, *)) {
                        for (NSString * name in [functions componentsSeparatedByString:@","]) {
                            MLModelConfiguration * function_config = [config copy];
                            function_config.functionName           = name;
                            MLModel *     function_model           = [MLModel modelWithContentsOfURL:url
                                                                                       configuration:function_config
                                                                                               error:&error];
                            CoreMLProgram program;
                            if (!function_model ||
                                !make_program(function_model, n_mels, 0, d_model, subsampling, capacity_out, program) ||
                                program.capacity >= main_program.capacity) {
                                log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "Core ML encoder: skipping function %s",
                                        name.UTF8String);
                                continue;
                            }
                            encoder->programs.push_back(program);
                        }
                        std::sort(
                            encoder->programs.begin(), encoder->programs.end(),
                            [](const CoreMLProgram & a, const CoreMLProgram & b) { return a.capacity < b.capacity; });
                    }
                }
                encoder->programs.push_back(main_program);
                encoder->n_mels       = n_mels;
                encoder->capacity     = main_program.capacity;
                encoder->d_model      = d_model;
                encoder->subsampling  = subsampling;
                encoder->capacity_out = main_program.capacity_out;
                log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "Core ML encoder loaded for %s (CPU + Neural Engine; GPU excluded)",
                        variant);
                return encoder.release();
            }
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "Core ML encoder: macOS 13 or later is required");
        } @catch (NSException * exception) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "Core ML encoder: %s", exception.reason.UTF8String);
        }
        return nullptr;
    }
}

void coreml_encoder_free(CoreMLEncoder * encoder) noexcept {
    @autoreleasepool {
        delete encoder;
    }
}

int coreml_encoder_capacity(const CoreMLEncoder * encoder) noexcept {
    return encoder->capacity;
}

int coreml_encoder_capacity_out(const CoreMLEncoder * encoder) noexcept {
    return encoder->capacity_out;
}

bool coreml_encoder_run(CoreMLEncoder *      encoder,
                        const float *        mel,
                        int                  n_frames,
                        bool                 time_major,
                        std::vector<float> & output,
                        int                  frames_out) {
    if (frames_out <= 0) {
        frames_out = (n_frames + encoder->subsampling - 1) / encoder->subsampling;
    }
    if (!mel || n_frames <= 0 || n_frames > encoder->capacity || (!encoder->length && n_frames != encoder->capacity) ||
        frames_out > encoder->capacity_out) {
        log_error("input length mismatch", nil);
        return false;
    }
    const CoreMLProgram & program =
        *std::find_if(encoder->programs.begin(), encoder->programs.end(),
                      [&](const CoreMLProgram & p) { return n_frames <= p.capacity && frames_out <= p.capacity_out; });
    @autoreleasepool {
        @try {
            float *         input    = static_cast<float *>(program.input.dataPointer);
            const NSInteger in_mel   = program.input.strides[1].integerValue;
            const NSInteger in_frame = program.input.strides[2].integerValue;
            for (int m = 0; m < encoder->n_mels; ++m) {
                for (int t = 0; t < program.capacity; ++t) {
                    input[m * in_mel + t * in_frame] =
                        t < n_frames ? mel[time_major ? t * encoder->n_mels + m : m * n_frames + t] : 0.0f;
                }
            }
            NSError *             error = nil;
            NSMutableDictionary * inputs =
                [@{ @"logmel_data" : [MLFeatureValue featureValueWithMultiArray:program.input] } mutableCopy];
            if (encoder->length) {
                *static_cast<int32_t *>(encoder->length.dataPointer) = n_frames;
                inputs[@"mel_length"] = [MLFeatureValue featureValueWithMultiArray:encoder->length];
            }
            MLDictionaryFeatureProvider * features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs
                                                                                                       error:&error];
            if (!features) {
                log_error("input features failed", error);
                return false;
            }
            id<MLFeatureProvider> prediction = [program.model predictionFromFeatures:features error:&error];
            MLMultiArray *        result     = [prediction featureValueForName:@"output"].multiArrayValue;
            if (!result || result.dataType != MLMultiArrayDataTypeFloat32 ||
                ![result.shape isEqualToArray:@[ @1, @(program.capacity_out), @(encoder->d_model) ]]) {
                log_error("prediction failed", error);
                return false;
            }
            output.resize(static_cast<size_t>(frames_out) * encoder->d_model);
            const float *   data      = static_cast<const float *>(result.dataPointer);
            const NSInteger out_frame = result.strides[1].integerValue;
            const NSInteger out_dim   = result.strides[2].integerValue;
            for (int t = 0; t < frames_out; ++t) {
                for (int d = 0; d < encoder->d_model; ++d) {
                    output[t * encoder->d_model + d] = data[t * out_frame + d * out_dim];
                }
            }
            return true;
        } @catch (NSException * exception) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "Core ML encoder: %s", exception.reason.UTF8String);
            return false;
        }
    }
}

}  // namespace transcribe
