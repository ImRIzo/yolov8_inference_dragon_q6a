#pragma once
#include <cstdint>
#include <vector>
#include <string>

/**
 * Initialize QNN runtime.
 * @param libDir  path to directory containing libQnnHtp.so, libQnnSystem.so, etc.
 * @returns true on success
 */
bool qnnInit(const std::string& libDir);

/**
 * Release all QNN resources.
 */
void qnnCleanup();

/**
 * Load a compiled context binary model and prepare for inference.
 * Populates tensor dimensions and quantization parameters from the binary.
 *
 * Output dequantization follows QNN's scale-offset encoding:
 *     float_value = (quantized_value + offset) * scale
 * The scale/offset pairs below come from the binary's tensor metadata
 * (authoritative), not from the config json.
 *
 * @param binPath       path to .bin context binary
 * @param cfgGraphName  graph name from config (may be empty to auto-detect)
 * @param outBoxScale   [out] dequantization scale for boxes output
 * @param outBoxOffset  [out] dequantization offset for boxes output
 * @param outScrScale   [out] dequantization scale for scores output
 * @param outScrOffset  [out] dequantization offset for scores output
 * @returns true on success
 */
bool loadModel(const std::string& binPath, const std::string& cfgGraphName,
               float& outBoxScale, int32_t& outBoxOffset,
               float& outScrScale, int32_t& outScrOffset);

/**
 * Run one inference pass.
 *
 * @param inputData   float32 NCHW [0,1] data (1 x 3 x 640 x 640)
 * @param outBoxBuf   [out] receives raw uint8 boxes output (caller must free)
 * @param outScrBuf   [out] receives raw uint8 scores output (caller must free)
 * @param boxDims     [out] dimensions of boxes tensor
 * @param scrDims     [out] dimensions of scores tensor
 * @returns true on success
 */
bool runInference(const std::vector<float>& inputData,
                  uint8_t*& outBoxBuf, uint32_t boxDims[4],
                  uint8_t*& outScrBuf, uint32_t scrDims[4]);

/**
 * Query the input/output tensor dimensions of the loaded model.
 * Used by the evaluation mode to validate the model I/O layout and to
 * print it for reproducibility.
 *
 * @param inDims   [out] input tensor dimensions (e.g. 1x3x640x640)
 * @param inRank   [out] input tensor rank
 * @param boxDims  [out] boxes output tensor dimensions (e.g. 1x4x8400)
 * @param boxRank  [out] boxes output tensor rank
 * @param scrDims  [out] scores output tensor dimensions (e.g. 1xNCx8400)
 * @param scrRank  [out] scores output tensor rank
 * @returns true if a model has been loaded and tensor info is available
 */
bool getModelShapes(uint32_t inDims[4], uint32_t& inRank,
                    uint32_t boxDims[4], uint32_t& boxRank,
                    uint32_t scrDims[4], uint32_t& scrRank);
