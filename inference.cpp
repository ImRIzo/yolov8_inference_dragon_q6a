#include "inference.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <dlfcn.h>

#include "QnnCommon.h"
#include "QnnTypes.h"
#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnDevice.h"
#include "QnnBackend.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

// ─── QNN function pointers (dlopen'd) ────────────────────────────────────────
typedef Qnn_ErrorHandle_t (*GetProvFn)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*GetSysFn)(const QnnSystemInterface_t***, uint32_t*);

static QnnBackend_CreateFn_t            pBC;
static QnnBackend_FreeFn_t              pBF;
static QnnDevice_CreateFn_t             pDC;
static QnnDevice_FreeFn_t               pDF;
static QnnContext_CreateFromBinaryFn_t  pCFB;
static QnnContext_FreeFn_t              pCF;
static QnnGraph_RetrieveFn_t            pGR;
static QnnGraph_ExecuteFn_t             pGE;
static QnnLog_CreateFn_t                pLC;
static QnnLog_FreeFn_t                  pLF;
static QnnProfile_CreateFn_t            pPC;
static QnnProfile_FreeFn_t              pPF;
static QnnSystemContext_CreateFn_t         pSC;
static QnnSystemContext_GetBinaryInfoFn_t  pSBI;
static QnnSystemContext_FreeFn_t           pSF;

// ─── Runtime handles ──────────────────────────────────────────────────────────
static void *g_hBe, *g_hSy;
static Qnn_BackendHandle_t  g_be;
static Qnn_DeviceHandle_t   g_dv;
static Qnn_ContextHandle_t  g_cx;
static Qnn_ProfileHandle_t  g_pf;
static Qnn_LogHandle_t      g_lg;
static Qnn_GraphHandle_t    g_gr;
static QnnSystemContext_Handle_t g_sc = nullptr;

// Model binary buffer (must outlive tensor metadata from system context)
static std::vector<uint8_t> g_mBuf;
static std::string g_graphName;

// ─── Tensor objects (built once after loadModel) ──────────────────────────────
static Qnn_Tensor_t g_inTensor, g_outBox, g_outScr;

// Tensor metadata extracted from binary introspection
struct TInfo {
    uint32_t id, dims[4], rank;
    Qnn_DataType_t dtype;
    Qnn_QuantizeParams_t qp;
};
static TInfo g_inInfo, g_boxInfo, g_scrInfo;
static bool g_inOK, g_boxOK, g_scrOK;

// ─── Helpers ──────────────────────────────────────────────────────────────────
static std::vector<uint8_t> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> b(sz);
    f.read((char*)b.data(), sz);
    return b;
}

// ─── Load function pointers ───────────────────────────────────────────────────
static bool loadQnnFunctions() {
    auto* gp = (GetProvFn)dlsym(g_hBe, "QnnInterface_getProviders");
    if (!gp) { fprintf(stderr, "dlsym QnnInterface_getProviders: %s\n", dlerror()); return false; }
    const QnnInterface_t** pv; uint32_t n;
    if (QNN_SUCCESS != gp(&pv, &n)) return false;
    bool ok = false;
    for (uint32_t i = 0; i < n; i++) {
        if (pv[i]->apiVersion.coreApiVersion.major == 2) {
            const auto& ifc = pv[i]->QNN_INTERFACE_VER_NAME;
            pBC  = ifc.backendCreate;            pBF  = ifc.backendFree;
            pDC  = ifc.deviceCreate;             pDF  = ifc.deviceFree;
            pCFB = ifc.contextCreateFromBinary;  pCF  = ifc.contextFree;
            pGR  = ifc.graphRetrieve;            pGE  = ifc.graphExecute;
            pLC  = ifc.logCreate;                pLF  = ifc.logFree;
            pPC  = ifc.profileCreate;            pPF  = ifc.profileFree;
            ok = true;
            printf("  QNN SDK v%d.%d.%d\n",
                   pv[i]->apiVersion.coreApiVersion.major,
                   pv[i]->apiVersion.coreApiVersion.minor,
                   pv[i]->apiVersion.coreApiVersion.patch);
            break;
        }
    }
    if (!ok) { fprintf(stderr, "ERROR: no matching QNN interface\n"); return false; }

    auto* gs = (GetSysFn)dlsym(g_hSy, "QnnSystemInterface_getProviders");
    if (!gs) return false;
    const QnnSystemInterface_t** sp; uint32_t sn;
    if (QNN_SUCCESS != gs(&sp, &sn)) return false;
    if (sn > 0) {
        const auto& si = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
        pSC  = si.systemContextCreate;
        pSBI = si.systemContextGetBinaryInfo;
        pSF  = si.systemContextFree;
    } else { fprintf(stderr, "ERROR: no QNN System interface\n"); return false; }
    return true;
}

// ─── Build tensor objects with correct IDs and quantisation params ────────────
static void buildTensors() {
    memset(&g_inTensor, 0, sizeof(g_inTensor));
    g_inTensor.version = QNN_TENSOR_VERSION_1;
    g_inTensor.v1.id = g_inInfo.id;
    g_inTensor.v1.name = (char*)"images";
    g_inTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    g_inTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    g_inTensor.v1.dataType = g_inInfo.dtype;
    g_inTensor.v1.quantizeParams = g_inInfo.qp;
    g_inTensor.v1.rank = g_inInfo.rank;
    g_inTensor.v1.dimensions = g_inInfo.dims;
    g_inTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;

    memset(&g_outBox, 0, sizeof(g_outBox));
    g_outBox.version = QNN_TENSOR_VERSION_1;
    g_outBox.v1.id = g_boxInfo.id;
    g_outBox.v1.name = (char*)"output_0";
    g_outBox.v1.type = QNN_TENSOR_TYPE_APP_READ;
    g_outBox.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    g_outBox.v1.dataType = g_boxInfo.dtype;
    g_outBox.v1.rank = g_boxInfo.rank;
    g_outBox.v1.dimensions = g_boxInfo.dims;
    g_outBox.v1.memType = QNN_TENSORMEMTYPE_RAW;

    memset(&g_outScr, 0, sizeof(g_outScr));
    g_outScr.version = QNN_TENSOR_VERSION_1;
    g_outScr.v1.id = g_scrInfo.id;
    g_outScr.v1.name = (char*)"output_1";
    g_outScr.v1.type = QNN_TENSOR_TYPE_APP_READ;
    g_outScr.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
    g_outScr.v1.dataType = g_scrInfo.dtype;
    g_outScr.v1.rank = g_scrInfo.rank;
    g_outScr.v1.dimensions = g_scrInfo.dims;
    g_outScr.v1.memType = QNN_TENSORMEMTYPE_RAW;
}

// ─── Allocate output tensor buffers ───────────────────────────────────────────
static void allocOutputBuffers() {
    for (auto* t : {&g_outBox, &g_outScr}) {
        size_t n = 1;
        for (uint32_t d = 0; d < t->v1.rank && t->v1.dimensions[d] > 0; d++) n *= t->v1.dimensions[d];
        size_t sz = n * (t->v1.dataType == QNN_DATATYPE_FLOAT_32
                         ? sizeof(float) : sizeof(uint8_t));
        t->v1.clientBuf.data = malloc(sz);
        t->v1.clientBuf.dataSize = (uint32_t)sz;
        memset(t->v1.clientBuf.data, 0, sz);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════════

bool qnnInit(const std::string& libDir) {
    g_hBe = dlopen((libDir + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_hBe) { fprintf(stderr, "dlopen HTP: %s\n", dlerror()); return false; }
    g_hSy = dlopen((libDir + "/libQnnSystem.so").c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_hSy) { fprintf(stderr, "dlopen Sys: %s\n", dlerror()); return false; }
    if (!loadQnnFunctions()) return false;
    if (pLC) pLC(nullptr, QNN_LOG_LEVEL_WARN, &g_lg);
    if (QNN_SUCCESS != pBC(g_lg, nullptr, &g_be)) {
        fprintf(stderr, "backendCreate failed\n"); return false;
    }
    if (pDC) {
        if (QNN_SUCCESS == pDC(g_lg, nullptr, &g_dv))
            printf("  Device OK\n");
        else g_dv = nullptr;
    }
    return true;
}

void qnnCleanup() {
    if (g_inTensor.v1.clientBuf.data) {
        free(g_inTensor.v1.clientBuf.data);
    }
    if (g_cx && pCF)  { pCF(g_cx, g_pf);  g_cx = nullptr; }
    if (g_dv && pDF)  { pDF(g_dv);         g_dv = nullptr; }
    if (g_pf && pPF)  { pPF(g_pf);         g_pf = nullptr; }
    if (g_be && pBF)  { pBF(g_be);         g_be = nullptr; }
    if (g_lg && pLF)  { pLF(g_lg);         g_lg = nullptr; }
    if (g_sc && pSF)  { pSF(g_sc);         g_sc = nullptr; }
    if (g_hSy)        { dlclose(g_hSy);   g_hSy = nullptr; }
    if (g_hBe)        { dlclose(g_hBe);   g_hBe = nullptr; }
}

bool loadModel(const std::string& binPath, const std::string& cfgGraphName,
               float& outBoxScale, int32_t& outBoxOffset,
               float& outScrScale, int32_t& outScrOffset) {
    g_mBuf = readFile(binPath);
    if (g_mBuf.empty()) return false;
    printf("  Model: %s (%zu bytes)\n", binPath.c_str(), g_mBuf.size());

    if (QNN_SUCCESS != pSC(&g_sc)) return false;
    const QnnSystemContext_BinaryInfo_t* bi = nullptr;
    Qnn_ContextBinarySize_t bsz = 0;
    if (QNN_SUCCESS != pSBI(g_sc, g_mBuf.data(), g_mBuf.size(), &bi, &bsz))
        return false;

    g_inOK = g_boxOK = g_scrOK = false;

    if (bi) {
        uint32_t ng = 0; QnnSystemContext_GraphInfo_t* gs = nullptr;
        if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
            { ng = bi->contextBinaryInfoV1.numGraphs; gs = bi->contextBinaryInfoV1.graphs; }
        else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
            { ng = bi->contextBinaryInfoV2.numGraphs; gs = bi->contextBinaryInfoV2.graphs; }
        else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
            { ng = bi->contextBinaryInfoV3.numGraphs; gs = bi->contextBinaryInfoV3.graphs; }

        for (uint32_t g = 0; g < ng && gs; g++) {
            auto& gi = gs[g];
            const char* gn = nullptr; uint32_t ni = 0, no = 0;
            Qnn_Tensor_t *it = nullptr, *ot = nullptr;

            if (gi.version == 1)
                { gn = gi.graphInfoV1.graphName; ni = gi.graphInfoV1.numGraphInputs;  it = gi.graphInfoV1.graphInputs;  no = gi.graphInfoV1.numGraphOutputs; ot = gi.graphInfoV1.graphOutputs; }
            else if (gi.version == 2)
                { gn = gi.graphInfoV2.graphName; ni = gi.graphInfoV2.numGraphInputs;  it = gi.graphInfoV2.graphInputs;  no = gi.graphInfoV2.numGraphOutputs; ot = gi.graphInfoV2.graphOutputs; }
            else if (gi.version == 3)
                { gn = gi.graphInfoV3.graphName; ni = gi.graphInfoV3.numGraphInputs;  it = gi.graphInfoV3.graphInputs;  no = gi.graphInfoV3.numGraphOutputs; ot = gi.graphInfoV3.graphOutputs; }

            if (cfgGraphName.empty() || (gn && cfgGraphName == gn))
                g_graphName = gn ? gn : "";

            printf("  Graph '%s' (%u in, %u out)\n", gn ? gn : "?", ni, no);

            for (uint32_t t = 0; t < ni && it; t++) {
                g_inInfo.id = it[t].v1.id;
                g_inInfo.rank = it[t].v1.rank;
                g_inInfo.dtype = it[t].v1.dataType;
                g_inInfo.qp = it[t].v1.quantizeParams;
                g_inOK = true;
                for (uint32_t d = 0; d < it[t].v1.rank && d < 4; d++)
                    g_inInfo.dims[d] = it[t].v1.dimensions[d];
            }
            for (uint32_t t = 0; t < no && ot; t++) {
                if (ot[t].v1.dimensions[1] == 4) {
                    g_boxInfo.id = ot[t].v1.id;
                    g_boxInfo.rank = ot[t].v1.rank;
                    g_boxInfo.dtype = ot[t].v1.dataType;
                    g_boxInfo.qp = ot[t].v1.quantizeParams;
                    g_boxOK = true;
                    for (uint32_t d = 0; d < ot[t].v1.rank && d < 4; d++)
                        g_boxInfo.dims[d] = ot[t].v1.dimensions[d];
                } else {
                    g_scrInfo.id = ot[t].v1.id;
                    g_scrInfo.rank = ot[t].v1.rank;
                    g_scrInfo.dtype = ot[t].v1.dataType;
                    g_scrInfo.qp = ot[t].v1.quantizeParams;
                    g_scrOK = true;
                    for (uint32_t d = 0; d < ot[t].v1.rank && d < 4; d++)
                        g_scrInfo.dims[d] = ot[t].v1.dimensions[d];
                }
            }
        }
    }
    // Keep g_sc alive — tensor metadata lives in system context

    if (g_graphName.empty() || !g_inOK || !g_boxOK || !g_scrOK) {
        fprintf(stderr, "ERROR: incomplete graph info\n"); return false;
    }

    if (QNN_SUCCESS != pCFB(g_be, g_dv, nullptr,
                            g_mBuf.data(), g_mBuf.size(), &g_cx, g_pf)) {
        fprintf(stderr, "ERROR: contextCreateFromBinary\n"); return false;
    }
    if (QNN_SUCCESS != pGR(g_cx, g_graphName.c_str(), &g_gr)) {
        fprintf(stderr, "ERROR: graphRetrieve\n"); return false;
    }

    // Dequantization parameters from the binary's tensor metadata:
    //   float_value = (quantized_value + offset) * scale
    if (g_boxOK &&
        g_boxInfo.qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        outBoxScale  = g_boxInfo.qp.scaleOffsetEncoding.scale;
        outBoxOffset = g_boxInfo.qp.scaleOffsetEncoding.offset;
    }
    if (g_scrOK &&
        g_scrInfo.qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        outScrScale  = g_scrInfo.qp.scaleOffsetEncoding.scale;
        outScrOffset = g_scrInfo.qp.scaleOffsetEncoding.offset;
    }

    buildTensors();
    allocOutputBuffers();

    printf("  Output dequant: boxes %.4f * (q + %d), scores %.6f * (q + %d)\n",
           outBoxScale, outBoxOffset, outScrScale, outScrOffset);
    printf("  Model loaded on HTP\n");
    return true;
}

bool runInference(const std::vector<float>& inputData,
                  uint8_t*& outBoxBuf, uint32_t boxDims[4],
                  uint8_t*& outScrBuf, uint32_t scrDims[4]) {
    // ── Fill input tensor ─────────────────────────────────────────────────
    size_t n = 1;
    for (uint32_t i = 0; i < g_inTensor.v1.rank; i++)
        n *= g_inTensor.v1.dimensions[i];

    // Model expects uint8 input.  Convert float32 [0,1] → uint8 [0,255].
    auto* inBuf = (uint8_t*)malloc(n);
    for (size_t i = 0; i < n; i++) {
        int v = (int)roundf(inputData[i] * 255.0f);
        inBuf[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    g_inTensor.v1.clientBuf.data = inBuf;
    g_inTensor.v1.clientBuf.dataSize = (uint32_t)n;

    // ── Execute ───────────────────────────────────────────────────────────
    Qnn_Tensor_t ins[]  = { g_inTensor };
    Qnn_Tensor_t ous[] = { g_outBox, g_outScr };
    auto rv = pGE(g_gr, ins, 1, ous, 2, g_pf, nullptr);
    if (QNN_SUCCESS != rv) {
        fprintf(stderr, "ERROR: graphExecute returned %lu\n", (unsigned long)rv);
        free(inBuf);
    g_inTensor.v1.clientBuf.data = nullptr;
        return false;
    }

    // ── Extract output buffers ────────────────────────────────────────────
    // graphExecute writes into the buffers we allocated; copy them out.
    size_t boxBytes = ous[0].v1.clientBuf.dataSize;
    size_t scrBytes = ous[1].v1.clientBuf.dataSize;

    outBoxBuf = (uint8_t*)malloc(boxBytes);
    outScrBuf = (uint8_t*)malloc(scrBytes);
    memcpy(outBoxBuf, ous[0].v1.clientBuf.data, boxBytes);
    memcpy(outScrBuf, ous[1].v1.clientBuf.data, scrBytes);

    for (int d = 0; d < 4; d++) {
        boxDims[d] = g_outBox.v1.dimensions[d];
        scrDims[d] = g_outScr.v1.dimensions[d];
    }

    free(inBuf);
    g_inTensor.v1.clientBuf.data = nullptr;
    return true;
}

bool getModelShapes(uint32_t inDims[4], uint32_t& inRank,
                    uint32_t boxDims[4], uint32_t& boxRank,
                    uint32_t scrDims[4], uint32_t& scrRank) {
    if (!g_inOK || !g_boxOK || !g_scrOK) return false;
    inRank = g_inInfo.rank;
    boxRank = g_boxInfo.rank;
    scrRank = g_scrInfo.rank;
    for (int d = 0; d < 4; d++) {
        inDims[d]  = g_inInfo.dims[d];
        boxDims[d] = g_boxInfo.dims[d];
        scrDims[d] = g_scrInfo.dims[d];
    }
    return true;
}
