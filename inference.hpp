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
 * Populates tensor dimensions and quantization scales from the binary.
 *
 * @param binPath       path to .bin context binary
 * @param cfgGraphName  graph name from config (may be empty to auto-detect)
 * @param outBoxScale   [out] dequantization scale for boxes output
 * @param outScrScale   [out] dequantization scale for scores output
 * @returns true on success
 */
bool loadModel(const std::string& binPath, const std::string& cfgGraphName,
               float& outBoxScale, float& outScrScale);

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
