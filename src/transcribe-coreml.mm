#include "transcribe-coreml.h"

#include "transcribe-log.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <limits>
#include <memory>

namespace transcribe {

struct CoreMLEncoder {
    MLModel *      model        = nil;
    MLMultiArray * input        = nil;
    MLMultiArray * length       = nil;
    int            n_mels       = 0;
    int            capacity     = 0;
    int            d_model      = 0;
    int            subsampling  = 0;
    int            capacity_out = 0;
};

static void log_error(const char * operation, NSError * error) {
    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "Core ML encoder: %s: %s", operation,
            error ? error.localizedDescription.UTF8String : "invalid model tensor contract");
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
                encoder->model                = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:model_path]
                                                                  configuration:config
                                                                          error:&error];
                if (!encoder->model) {
                    log_error("load failed", error);
                    return nullptr;
                }
                MLModelDescription * desc           = encoder->model.modelDescription;
                NSString *           source_variant = desc.metadata[MLModelCreatorDefinedKey][@"transcribe.variant"];
                if (![source_variant isEqualToString:[NSString stringWithUTF8String:variant]]) {
                    log_error("encoder checkpoint does not match GGUF variant", nil);
                    return nullptr;
                }
                MLMultiArrayConstraint * input  = desc.inputDescriptionsByName[@"logmel_data"].multiArrayConstraint;
                MLMultiArrayConstraint * output = desc.outputDescriptionsByName[@"output"].multiArrayConstraint;
                if (!input || input.shape.count != 3 || input.shape[0].intValue != 1 ||
                    input.shape[1].intValue != n_mels || input.shape[2].longLongValue <= 0 ||
                    input.shape[2].longLongValue > std::numeric_limits<int>::max() - subsampling ||
                    (capacity > 0 && input.shape[2].intValue != capacity)) {
                    log_error("input shape mismatch", nil);
                    return nullptr;
                }
                capacity = input.shape[2].intValue;
                if (capacity_out == 0) {
                    capacity_out = (capacity + subsampling - 1) / subsampling;
                } else if (capacity_out < 0 && output && output.shape.count == 3) {
                    capacity_out = output.shape[1].intValue;
                }
                if (!output || capacity_out <= 0 ||
                    ![output.shape isEqualToArray:@[ @1, @(capacity_out), @(d_model) ]] ||
                    input.dataType != MLMultiArrayDataTypeFloat32 || output.dataType != MLMultiArrayDataTypeFloat32) {
                    log_error("output shape or dtype mismatch", nil);
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
                encoder->input = [[MLMultiArray alloc] initWithShape:input.shape
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                               error:&error];
                if (!encoder->input) {
                    log_error("input allocation failed", error);
                    return nullptr;
                }
                encoder->n_mels       = n_mels;
                encoder->capacity     = capacity;
                encoder->d_model      = d_model;
                encoder->subsampling  = subsampling;
                encoder->capacity_out = capacity_out;
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
    @autoreleasepool {
        @try {
            float *    input   = static_cast<float *>(encoder->input.dataPointer);
            const auto strides = encoder->input.strides;
            for (int m = 0; m < encoder->n_mels; ++m) {
                for (int t = 0; t < encoder->capacity; ++t) {
                    input[m * strides[1].integerValue + t * strides[2].integerValue] =
                        t < n_frames ? mel[time_major ? t * encoder->n_mels + m : m * n_frames + t] : 0.0f;
                }
            }
            NSError *             error = nil;
            NSMutableDictionary * inputs =
                [@{ @"logmel_data" : [MLFeatureValue featureValueWithMultiArray:encoder->input] } mutableCopy];
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
            id<MLFeatureProvider> prediction = [encoder->model predictionFromFeatures:features error:&error];
            MLMultiArray *        result     = [prediction featureValueForName:@"output"].multiArrayValue;
            if (!result || result.dataType != MLMultiArrayDataTypeFloat32 ||
                ![result.shape isEqualToArray:@[ @1, @(encoder->capacity_out), @(encoder->d_model) ]]) {
                log_error("prediction failed", error);
                return false;
            }
            output.resize(static_cast<size_t>(frames_out) * encoder->d_model);
            const float * data        = static_cast<const float *>(result.dataPointer);
            const auto    out_strides = result.strides;
            for (int t = 0; t < frames_out; ++t) {
                for (int d = 0; d < encoder->d_model; ++d) {
                    output[t * encoder->d_model + d] =
                        data[t * out_strides[1].integerValue + d * out_strides[2].integerValue];
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
