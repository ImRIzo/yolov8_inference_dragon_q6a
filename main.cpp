/**
 * YOLOv8 Video Inference — Radxa Dragon Q6A NPU (QCS6490)
 *
 * Reads a video file (or camera), runs YOLOv8 on every frame via QNN HTP,
 * displays live output with FPS, timing, and detection overlays.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "preprocess.hpp"
#include "inference.hpp"
#include "postprocess.hpp"
#include "evaluate.hpp"

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static constexpr float DEFAULT_CONF = 0.25f;
static constexpr float DEFAULT_IOU  = 0.50f;
static constexpr int   WINDOW_W     = 650;
static constexpr int   WINDOW_H     = 650;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<uint8_t> readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> b(sz);
    f.read((char*)b.data(), sz);
    return b;
}

static bool readConfig(const std::string& path,
                       std::string& graphName,
                       float& boxScale, float& scrScale) {
    auto d = readFile(path);
    if (d.empty()) return false;
    std::string c((const char*)d.data(), d.size());
    auto es = [&](const std::string& k) -> std::string {
        size_t p = c.find("\"" + k + "\"");
        if (p == std::string::npos) return "";
        p = c.find(':', p); if (p == std::string::npos) return "";
        p = c.find('"', p); if (p == std::string::npos) return "";
        size_t e = c.find('"', p + 1);
        if (e == std::string::npos) return "";
        return c.substr(p + 1, e - p - 1);
    };
    auto ef = [&](const std::string& l) -> float {
        size_t p = c.find("\"" + l + "\"");
        if (p == std::string::npos) return 0;
        p = c.find("scale", p); if (p == std::string::npos) return 0;
        p = c.find(':', p); if (p == std::string::npos) return 0;
        while (p + 1 < c.size() && (c[p + 1] == ' ' || c[p + 1] == '\t')) p++;
        return strtof(c.c_str() + p + 1, nullptr);
    };
    graphName = es("graph_name");
    float b0 = ef("output_0"), s0 = ef("output_1");
    if (b0 > 0) boxScale = b0;
    if (s0 > 0) scrScale = s0;
    return true;
}

static bool findModel(const std::string& exeDir,
                      std::string& bin, std::string& cfg) {
    std::string md = exeDir + "model";
    const char* ds[] = { md.c_str(),
        "../quantized_compiled_model", "../../quantized_compiled_model", nullptr };
    for (int i = 0; ds[i]; i++) {
        std::string cmd = std::string("ls ") + ds[i] + "/*.bin 2>/dev/null | tail -1";
        FILE* fp = popen(cmd.c_str(), "r");
        if (!fp) continue;
        char ln[1024] = {};
        if (fgets(ln, sizeof(ln), fp)) {
            ln[strcspn(ln, "\n")] = 0; pclose(fp);
            if (strlen(ln) > 0) {
                bin = ln; cfg = bin;
                size_t p = cfg.rfind(".bin");
                if (p != std::string::npos) { cfg.replace(p, 4, "_config.json"); return true; }
            }
        } else { pclose(fp); }
    }
    return false;
}

// ─── Resize frame for display while keeping aspect ratio ─────────────────────
static cv::Mat resizeDisplay(const cv::Mat& frame, int maxW, int maxH) {
    double scale = std::min((double)maxW / frame.cols, (double)maxH / frame.rows);
    if (scale >= 1.0) return frame;
    cv::Mat out;
    cv::resize(frame, out, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    printf("============================================================\n"
           "  YOLOv8 Video — Radxa Q6A NPU — C++\n"
           "============================================================\n");

    // ── Locate ourselves ──────────────────────────────────────────────────
    std::string exeDir = "./";
    {
        char buf[4096] = {};
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) { buf[n] = 0; exeDir = buf; exeDir = exeDir.substr(0, exeDir.rfind('/') + 1); }
    }

    std::string libDir   = exeDir + "lib";
    std::string videoPath = exeDir + "test_video.mp4";
    bool isRtsp = false;
    float confTh = DEFAULT_CONF, iouTh = DEFAULT_IOU;
    bool showOutput = true;
    bool confGiven = false, iouGiven = false;
    bool evalMode = false;                    // static labeled-image evaluation mode
    std::string evalDataDir = "data/valid";   // dataset dir for legacy --map
    std::string imagesDir, labelsDir;         // explicit --images / --labels dirs
    int numClasses = 1;
    int maxDet = 300;                         // evaluation max detections per image
    int warmupFrames = 5;                     // warm-up inferences before timing
    std::string outDir;                       // output dir (default: exe dir)
    bool autoPython = true;                   // auto-run evaluate_metrics.py

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--video")  && i+1 < argc) videoPath = argv[++i];
        else if (!strcmp(argv[i], "--rtsp")   && i+1 < argc) { videoPath = argv[++i]; isRtsp = true; }
        else if (!strcmp(argv[i], "--lib")    && i+1 < argc) libDir    = argv[++i];
        else if (!strcmp(argv[i], "--conf")   && i+1 < argc) { confTh    = strtof(argv[++i], nullptr); confGiven = true; }
        else if (!strcmp(argv[i], "--iou")    && i+1 < argc) { iouTh     = strtof(argv[++i], nullptr); iouGiven  = true; }
        else if (!strcmp(argv[i], "--classes") && i+1 < argc) numClasses = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-det") && i+1 < argc) maxDet     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i+1 < argc) warmupFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--images")  && i+1 < argc) imagesDir  = argv[++i];
        else if (!strcmp(argv[i], "--labels")  && i+1 < argc) labelsDir  = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) outDir     = argv[++i];
        else if (!strcmp(argv[i], "--eval"))                  evalMode = true;
        else if (!strcmp(argv[i], "--no-python-eval"))        autoPython = false;
        else if (!strcmp(argv[i], "--map")) {
            evalMode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') evalDataDir = argv[++i];
        }
        else if (!strcmp(argv[i], "--no-show"))               showOutput = false;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--video PATH] [--rtsp URL] [--lib DIR] [--conf FLOAT] [--iou FLOAT]\n"
                   "             [--no-show] [--map [DATADIR]] [--classes N] [--help]\n"
                   "             [--eval --images DIR --labels DIR] [--max-det N] [--warmup N] [--out DIR]\n"
                   "\n"
                   "  Video file:\n"
                   "    %s --video test_video.mp4\n"
                   "  RTSP stream:\n"
                   "    %s --rtsp rtsp://192.168.1.100:8554/stream\n"
                   "\n"
                   "  Static image evaluation (labeled test images, YOLO txt labels):\n"
                   "    %s --eval --images data/test/images --labels data/test/labels --no-show\n"
                   "    %s --eval --images data/test/images   (labels dir defaults to ../labels)\n"
                   "  Evaluation defaults (Ultralytics val.py): conf 0.001, NMS IoU 0.70, max det 300.\n"
                   "  Raw detections are saved to <outdir>/predictions.csv; Ultralytics-exact metrics\n"
                   "  are computed by evaluate_metrics.py and saved to <outdir>/evaluation_results.csv.\n"
                   "\n"
                   "  Legacy mAP dataset mode (images/ + labels/ inside DATADIR):\n"
                   "    %s --map data/test --classes 1 --conf 0.001 --iou 0.7\n",
                   argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        }
    }

    // --images implies evaluation mode
    if (!imagesDir.empty()) evalMode = true;

    if (evalMode)
        printf("\n  Lib:         %s\n  Images dir:  %s\n  Labels dir:  %s\n\n",
               libDir.c_str(), imagesDir.empty() ? evalDataDir.c_str() : imagesDir.c_str(),
               labelsDir.empty() ? "(default)" : labelsDir.c_str());
    else
        printf("\n  Video:  %s\n  Lib:    %s\n  Conf:   %.2f  IoU: %.2f\n\n",
               videoPath.c_str(), libDir.c_str(), confTh, iouTh);

    // ── ADSP ──────────────────────────────────────────────────────────────
    setenv("ADSP_LIBRARY_PATH", libDir.c_str(), 1);

    // ── Find model ────────────────────────────────────────────────────────
    std::string binPath, cfgPath;
    if (!findModel(exeDir, binPath, cfgPath)) {
        fprintf(stderr, "ERROR: no .bin model found\n"); return 1;
    }
    std::string graphName;
    float boxScale = 2.5563f, scrScale = 0.0038f;   // fallbacks only
    int32_t boxOffset = 0, scrOffset = 0;           // from model metadata
    readConfig(cfgPath, graphName, boxScale, scrScale);
    printf("Model: %s\n\n", binPath.c_str());

    // ── Init QNN (once) ───────────────────────────────────────────────────
    printf("--- QNN Init ---\n");
    if (!qnnInit(libDir)) return 1;

    printf("\n--- Load Model ---\n");
    if (!loadModel(binPath, graphName, boxScale, boxOffset, scrScale, scrOffset)) {
        qnnCleanup(); return 1;
    }

    // ── Static labeled-image evaluation mode (--eval / --images / --map) ──
    if (evalMode) {
        // Ultralytics val.py defaults unless the user overrides via --conf/--iou.
        // (The live video demo keeps its own 0.25 / 0.50 thresholds.)
        if (!confGiven) confTh = 0.001f;
        if (!iouGiven)  iouTh  = 0.7f;
        EvalOptions eo;
        eo.imagesDir    = imagesDir;
        eo.labelsDir    = labelsDir;
        eo.dataDir      = imagesDir.empty() ? evalDataDir : "";   // legacy --map
        eo.outDir       = outDir.empty() ? exeDir : outDir;
        eo.modelPath    = binPath;
        eo.runtimeName  = "INT8 QNN HTP NPU (QCS6490)";
        eo.numClasses   = numClasses;
        eo.confThresh   = confTh;
        eo.iouThresh    = iouTh;
        eo.maxDet       = maxDet;
        eo.warmupFrames = warmupFrames;
        eo.autoPython   = autoPython;
        // Locate evaluate_metrics.py: next to the binary, or in the repo root
        // when running from the build directory.
        eo.pythonScript = exeDir + "evaluate_metrics.py";
        {
            FILE* f = fopen(eo.pythonScript.c_str(), "r");
            if (f) fclose(f);
            else {
                std::string alt = exeDir + "../evaluate_metrics.py";
                FILE* f2 = fopen(alt.c_str(), "r");
                if (f2) { fclose(f2); eo.pythonScript = alt; }
            }
        }
        bool ok = runEvaluation(eo, boxScale, boxOffset, scrScale, scrOffset);
        qnnCleanup();
        return ok ? 0 : 1;
    }

    // ── Open video ────────────────────────────────────────────────────────
    // RTSP: reduce buffer for lower latency, set timeout
    if (isRtsp) {
    }
    cv::VideoCapture cap(videoPath);
    if (isRtsp) {
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);       // minimal buffering for low latency
        cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); // 5 sec connection timeout
        printf("  RTSP stream mode (low-latency)\n");
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "ERROR: cannot open video: %s\n", videoPath.c_str());
        qnnCleanup(); return 1;
    }

    double videoFps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    printf("\n--- Video ---\n");
    printf("  %s\n", videoPath.c_str());
    printf("  %.1f fps, %d frames\n\n", videoFps, totalFrames);

    // ── Display window ────────────────────────────────────────────────────
    if (showOutput) {
        cv::namedWindow("YOLOv8 — Radxa Q6A NPU", cv::WINDOW_NORMAL);
        cv::resizeWindow("YOLOv8 — Radxa Q6A NPU", WINDOW_W, WINDOW_H);
    }

    // ── Per-frame timing accumulators ─────────────────────────────────────
    double totalPreMs = 0, totalInfMs = 0, totalPostMs = 0;
    int frameCount = 0;
    double smoothFps = 0.0;       // EMA-smoothed wall-clock FPS for display
    double minFps = 1e9, maxFps = 0.0;  // min/max wall-clock FPS (after warmup)
    const int WARMUP_FRAMES = 5;
    auto videoStart = Clock::now();
    auto lastFrameTime = videoStart;   // for wall-clock frame-to-frame FPS

    cv::Mat rawFrame;
    int reconnectAttempts = 0;
    const int MAX_RECONNECT = 10;
    while (cap.read(rawFrame)) {
        // RTSP reconnect on empty frame
        if (isRtsp && rawFrame.empty()) {
            if (++reconnectAttempts > MAX_RECONNECT) {
                fprintf(stderr, "RTSP: max reconnects reached, exiting\n");
                break;
            }
            fprintf(stderr, "RTSP: frame dropped, reconnecting (%d/%d)...\n", reconnectAttempts, MAX_RECONNECT);
            cap.release();
            sleep(1);
            if (!cap.open(videoPath)) { fprintf(stderr, "RTSP: reconnect failed\n"); break; }
            continue;
        }
        reconnectAttempts = 0;
        if (rawFrame.empty()) break;

        // ── Preprocess ────────────────────────────────────────────────────
        auto t0 = Clock::now();
        auto pre = preprocess(rawFrame);
        auto t1 = Clock::now();
        double preMs = Ms(t1 - t0).count();

        if (pre.data.empty()) continue;

        // ── Inference ─────────────────────────────────────────────────────
        auto t2 = Clock::now();
        uint8_t *boxBuf = nullptr, *scrBuf = nullptr;
        uint32_t boxDims[4], scrDims[4];
        bool ok = runInference(pre.data, boxBuf, boxDims, scrBuf, scrDims);
        auto t3 = Clock::now();
        double infMs = Ms(t3 - t2).count();

        if (!ok) { qnnCleanup(); return 1; }

        // ── Postprocess ───────────────────────────────────────────────────
        auto t4 = Clock::now();
        auto dets = parseOutputs(boxBuf, scrBuf, boxDims, scrDims,
                                  boxScale, boxOffset, scrScale, scrOffset, confTh);
        auto nd = nms(dets, confTh, iouTh);
        auto result = drawDetections(rawFrame, nd, COCO_CLASSES,
                                      pre.scale, pre.padLeft, pre.padTop);
        auto t5 = Clock::now();
        double postMs = Ms(t5 - t4).count();

        free(boxBuf);
        free(scrBuf);

        // ── Accumulate ────────────────────────────────────────────────────
        totalPreMs  += preMs;
        totalInfMs  += infMs;
        totalPostMs += postMs;
        frameCount++;

        double elapsedSec = Ms(Clock::now() - videoStart).count() / 1000.0;
        double avgFps = (elapsedSec > 0) ? (frameCount / elapsedSec) : 0;

        // ── Wall-clock frame-to-frame FPS (includes display overhead) ─────
        auto now = Clock::now();
        double wallMs = Ms(now - lastFrameTime).count();
        lastFrameTime = now;
        double wallFps = (wallMs > 0) ? (1000.0 / wallMs) : 0;

        // EMA smoothing — dampens the jumping FPS counter
        if (smoothFps <= 0.0)
            smoothFps = wallFps;
        else
            smoothFps = 0.15 * wallFps + 0.85 * smoothFps;

        // Track min/max wall-clock FPS (skip warmup frames, cap outliers)
        if (frameCount > WARMUP_FRAMES) {
            if (wallFps < minFps) minFps = wallFps;
            if (wallFps > maxFps) maxFps = wallFps;
        }

        // ── Display ───────────────────────────────────────────────────────
        if (showOutput) {
            // 1. Resize detection result for the display window
            cv::Mat display = resizeDisplay(result, WINDOW_W, WINDOW_H);

            // 2. Draw overlay ON the display-sized frame so text stays
            //    clear and sharp regardless of source video resolution.
            drawDebugOverlay(display, preMs, infMs, postMs,
                             smoothFps, avgFps, minFps, maxFps,
                             frameCount, (int)nd.size());

            cv::imshow("YOLOv8 — Radxa Q6A NPU", display);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break;  // ESC or q
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    double totalSec = Ms(Clock::now() - videoStart).count() / 1000.0;
    double avgFpsFinal = (totalSec > 0) ? (frameCount / totalSec) : 0;
    printf("\n============================================================\n");
    printf("  Processed %d frames in %.1f s\n", frameCount, totalSec);
    printf("  Average FPS:        %.1f\n", avgFpsFinal);
    printf("  Minimum FPS:        %.1f\n", (frameCount > WARMUP_FRAMES) ? minFps : avgFpsFinal);
    printf("  Maximum FPS:        %.1f\n", (frameCount > WARMUP_FRAMES) ? maxFps : avgFpsFinal);
    printf("  Avg preprocess:     %.2f ms\n", totalPreMs  / frameCount);
    printf("  Avg inference:      %.2f ms\n", totalInfMs  / frameCount);
    printf("  Avg postprocess:    %.2f ms\n", totalPostMs / frameCount);
    printf("  Avg total:          %.2f ms\n",
           (totalPreMs + totalInfMs + totalPostMs) / frameCount);
    printf("============================================================\n");

    // ── Write report ──────────────────────────────────────────────────────
    {
        std::string reportPath = exeDir + "report.txt";
        FILE* rpt = fopen(reportPath.c_str(), "w");
        if (rpt) {
            time_t now = time(nullptr);
            double avgPre  = frameCount ? (totalPreMs  / frameCount) : 0;
            double avgInf  = frameCount ? (totalInfMs  / frameCount) : 0;
            double avgPost = frameCount ? (totalPostMs / frameCount) : 0;
            double avgTot  = frameCount ? ((totalPreMs + totalInfMs + totalPostMs) / frameCount) : 0;
            fprintf(rpt, "============================================================\n");
            fprintf(rpt, "  YOLOv8 Video Inference Report\n");
            fprintf(rpt, "============================================================\n");
            fprintf(rpt, "  Date:       %s", ctime(&now));
            fprintf(rpt, "  Video:      %s\n", videoPath.c_str());
            fprintf(rpt, "  Model:      %s\n", binPath.c_str());
            fprintf(rpt, "  Confidence: %.2f\n", confTh);
            fprintf(rpt, "  IoU:        %.2f\n", iouTh);
            fprintf(rpt, "------------------------------------------------------------\n");
            fprintf(rpt, "  Frames:     %d\n", frameCount);
            fprintf(rpt, "  Total time: %.2f s\n", totalSec);
            fprintf(rpt, "  Avg FPS:    %.1f\n", avgFpsFinal);
            if (frameCount > WARMUP_FRAMES) {
                fprintf(rpt, "  Min FPS:    %.1f\n", minFps);
                fprintf(rpt, "  Max FPS:    %.1f\n", maxFps);
            }
            fprintf(rpt, "------------------------------------------------------------\n");
            fprintf(rpt, "  Avg preprocess:   %.2f ms\n", avgPre);
            fprintf(rpt, "  Avg inference:    %.2f ms\n", avgInf);
            fprintf(rpt, "  Avg postprocess:  %.2f ms\n", avgPost);
            fprintf(rpt, "  Avg total:        %.2f ms\n", avgTot);
            fprintf(rpt, "============================================================\n");
            fclose(rpt);
            printf("\nReport saved: %s\n", reportPath.c_str());
        }
    }
    cap.release();
    if (showOutput) cv::destroyAllWindows();
    qnnCleanup();
    return 0;
}
