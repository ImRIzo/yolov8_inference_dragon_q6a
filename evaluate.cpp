/**
 * evaluate.cpp — COCO/Ultralytics-style mAP evaluation (mAP@0.5, mAP@0.5:0.95, P/R/F1)
 *
 * Walks a dataset (images/ + labels/ in YOLO txt format, or separate
 * --images/--labels directories), runs the full QNN pipeline on every image
 * and computes:
 *
 *   - mAP@0.5        AP at IoU threshold 0.5
 *   - mAP@0.5:0.95   mean AP over 10 IoU thresholds (0.50 … 0.95, step 0.05)
 *   - Precision / Recall / F1-score at the max-F1 operating point
 *
 * AP is computed from 101-point interpolated precision-recall curves, the
 * same convention as Ultralytics / COCO.  P/R/F1 curves are sampled on 1000
 * confidence bins and the operating point is the bin where the smoothed
 * mean-F1 curve peaks (Ultralytics-style).
 *
 * RAW DETECTIONS (after dequantization → YOLO decode → reverse letterbox →
 * NMS → max-det cap) are written to <outDir>/predictions.csv together with
 * audit files (<outDir>/image_sizes.csv, <outDir>/eval_config.csv,
 * <outDir>/evaluated_images.txt).  The companion script evaluate_metrics.py
 * reproduces the exact Ultralytics val.py computation from predictions.csv —
 * use its numbers for the FP32-vs-INT8 research comparison.
 *
 * Evaluation settings default to Ultralytics val.py values:
 *   conf = 0.001, NMS IoU = 0.70, max_det = 300
 * (the live video demo keeps its own 0.25 / 0.50 thresholds).
 *
 * Requires qnnInit() + loadModel() to have been called before use.
 */
#include "evaluate.hpp"
#include "preprocess.hpp"
#include "inference.hpp"
#include "postprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

namespace {

constexpr int   NUM_IOU_THRESH = 10;    // IoU thresholds 0.50 … 0.95
constexpr int   NBINS          = 1000;  // confidence bins for the P/R/F1 curves
const char*     IMG_EXTS[]     = { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };

struct GtBox { int cls; float x1, y1, x2, y2; };

struct EvalImage {
    std::string name;
    int w = 0, h = 0;
    std::vector<GtBox> gt;
    std::vector<Detection> preds;  // pixel coordinates
};

struct ScoredMatch {
    float conf;
    int cls;
    bool tp;
};

// ─── Small helpers ─────────────────────────────────────────────────────────────

static std::string fmt(const char* format, ...) {
    char buf[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    return std::string(buf);
}

static const char* className(int c) {
    static char fallback[32];
    if (c >= 0 && c < 80 && COCO_CLASSES[c]) return COCO_CLASSES[c];
    snprintf(fallback, sizeof(fallback), "class_%d", c);
    return fallback;
}

static bool hasExt(const std::string& s, const char* ext) {
    size_t n = s.size(), m = strlen(ext);
    if (n < m) return false;
    for (size_t i = 0; i < m; i++)
        if ((char)tolower((unsigned char)s[n - m + i]) != ext[i]) return false;
    return true;
}

static bool isSupportedImage(const std::string& name) {
    for (const char* ext : IMG_EXTS)
        if (hasExt(name, ext)) return true;
    return false;
}

static bool dirExists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string stemOf(const std::string& name) {
    size_t p = name.rfind('.');
    return (p == std::string::npos) ? name : name.substr(0, p);
}

static std::string baseName(const std::string& p) {
    std::string s = p;
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    size_t sl = s.rfind('/');
    return (sl == std::string::npos) ? s : s.substr(sl + 1);
}

static std::string parentDir(const std::string& p) {
    std::string s = p;
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    size_t sl = s.rfind('/');
    if (sl == std::string::npos) return ".";
    if (sl == 0) return "/";
    return s.substr(0, sl);
}

// List supported images in a directory; other (non-dot) files are collected
// into `unsupported` so the caller can warn about them.
static std::vector<std::string> listImages(const std::string& dir,
                                           std::vector<std::string>& unsupported) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        if (name == "." || name == ".." || name[0] == '.') continue;
        if (isSupportedImage(name)) out.push_back(name);
        else {
            // skip directories; warn about regular files with other extensions
            struct stat st;
            std::string full = dir + "/" + name;
            if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode))
                unsupported.push_back(name);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// "1x3x640x640" style shape string for a rank 1..4 tensor.
static std::string shapeStr(const uint32_t dims[4], uint32_t rank) {
    uint32_t r = (rank >= 1 && rank <= 4) ? rank : 4;
    std::string s;
    for (uint32_t d = 0; d < r; d++) {
        if (d) s += 'x';
        s += fmt("%u", dims[d]);
    }
    return s;
}

// Trim leading/trailing whitespace.
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Mean / population standard deviation / min / max (np.std(ddof=0) convention).
struct TimeStats { double mean = 0, sd = 0, mn = 0, mx = 0; };
static TimeStats timeStats(const std::vector<double>& v) {
    TimeStats s;
    if (v.empty()) return s;
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / (double)v.size();
    double ss = 0;
    s.mn = s.mx = v[0];
    for (double x : v) {
        ss += (x - s.mean) * (x - s.mean);
        if (x < s.mn) s.mn = x;
        if (x > s.mx) s.mx = x;
    }
    s.sd = std::sqrt(ss / (double)v.size());
    return s;
}

static std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// Best-effort: run evaluate_metrics.py for Ultralytics-exact metrics.
static void runPythonMetrics(const EvalOptions& opt, const std::string& predCsv,
                             const std::string& lblDir, const std::string& cfgCsv,
                             const std::string& outCsv) {
    if (opt.pythonScript.empty()) {
        printf("\nNOTE: evaluate_metrics.py not located — run it manually for\n"
               "      Ultralytics-exact metrics (see README).\n");
        return;
    }
    if (access(opt.pythonScript.c_str(), R_OK) != 0) {
        printf("\nNOTE: evaluate_metrics.py not found at %s — run it manually.\n",
               opt.pythonScript.c_str());
        return;
    }
    const char* pyEnv = getenv("QNN_EVAL_PYTHON");
    std::string interp = (pyEnv && pyEnv[0]) ? pyEnv : "python3";
    if (system((interp + " -c \"import numpy\" >/dev/null 2>&1").c_str()) != 0) {
        printf("\nNOTE: '%s' has no numpy module — evaluate_metrics.py was not run.\n"
               "  Install numpy (or set QNN_EVAL_PYTHON to a python with numpy) and run:\n"
               "  %s %s --predictions %s --labels %s --config %s --output %s\n",
               interp.c_str(), interp.c_str(), opt.pythonScript.c_str(),
               predCsv.c_str(), lblDir.c_str(), cfgCsv.c_str(), outCsv.c_str());
        return;
    }
    std::string cmd = shellQuote(interp) + " " + shellQuote(opt.pythonScript) +
                      " --predictions " + shellQuote(predCsv) +
                      " --labels "     + shellQuote(lblDir) +
                      " --config "     + shellQuote(cfgCsv) +
                      " --output "     + shellQuote(outCsv);
    printf("\n--- Ultralytics-exact metrics (evaluate_metrics.py) ---\n");
    int rc = system(cmd.c_str());
    if (rc != 0)
        printf("\nNOTE: evaluate_metrics.py exited with code %d — rerun manually:\n  %s\n",
               rc, cmd.c_str());
}

// ─── IoU ───────────────────────────────────────────────────────────────────────

static float boxIoU(const Detection& a, const GtBox& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, x2 - x1), ih = std::max(0.0f, y2 - y1);
    float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;
    float uni = (a.x2 - a.x1) * (a.y2 - a.y1) +
                (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// ─── 101-point interpolated AP (equivalent of np.trapz(np.interp(x, mrec, mpre))) ───

static float computeAP101(const std::vector<float>& recall,
                          const std::vector<float>& precision) {
    std::vector<float> mrec, mpre;
    mrec.reserve(recall.size() + 2);
    mpre.reserve(precision.size() + 2);
    mrec.push_back(0.0f); mpre.push_back(1.0f);
    mrec.insert(mrec.end(), recall.begin(), recall.end());
    mpre.insert(mpre.end(), precision.begin(), precision.end());
    mrec.push_back(1.0f); mpre.push_back(0.0f);

    for (int i = (int)mpre.size() - 2; i >= 1; i--)   // precision envelope
        mpre[i] = std::max(mpre[i], mpre[i + 1]);

    // Trapezoidal integration over the 101 sample points (np.trapz convention:
    // endpoint samples get half weight).
    double ap = 0.0;
    for (int k = 0; k < 101; k++) {
        double x = k / 100.0;
        size_t i = std::upper_bound(mrec.begin(), mrec.end(), (float)x) - mrec.begin();
        if (i == 0) i = 1;
        if (i >= mrec.size()) i = mrec.size() - 1;
        double x0 = mrec[i - 1], x1 = mrec[i];
        double y0 = mpre[i - 1], y1 = mpre[i];
        double y = (x1 > x0) ? y0 + (x - x0) * (y1 - y0) / (x1 - x0) : y1;
        double w = (k == 0 || k == 100) ? 0.005 : 0.01;
        ap += y * w;
    }
    return (float)ap;
}

// ─── Greedy per-image matching ─────────────────────────────────────────────────
// Sorts predictions by confidence (desc) and matches each against the best
// unused GT box of the same class with IoU > thr.

static std::vector<ScoredMatch> matchImage(const EvalImage& img, float thr) {
    std::vector<int> order(img.preds.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return img.preds[a].confidence > img.preds[b].confidence;
    });

    std::vector<bool> used(img.gt.size(), false);
    std::vector<ScoredMatch> out;
    out.reserve(order.size());
    for (int oi : order) {
        const Detection& p = img.preds[oi];
        int best = -1;
        float bestIoU = thr;
        for (size_t g = 0; g < img.gt.size(); g++) {
            if (used[g] || img.gt[g].cls != p.classId) continue;
            float iou = boxIoU(p, img.gt[g]);
            if (iou > bestIoU) { bestIoU = iou; best = (int)g; }
        }
        bool tp = false;
        if (best >= 0) { used[best] = true; tp = true; }
        out.push_back({ p.confidence, p.classId, tp });
    }
    return out;
}

// Match every image at one IoU threshold and gather per-class,
// confidence-sorted (desc) TP/FP lists.
static void gatherMatches(const std::vector<EvalImage>& images, float thr,
                          int numClasses,
                          std::vector<std::vector<ScoredMatch>>& perClass) {
    perClass.assign((size_t)numClasses, std::vector<ScoredMatch>());
    for (const EvalImage& img : images) {
        std::vector<ScoredMatch> m = matchImage(img, thr);
        for (const ScoredMatch& s : m) {
            if (s.cls < 0 || s.cls >= numClasses) continue;
            perClass[s.cls].push_back(s);
        }
    }
    for (auto& v : perClass)
        std::stable_sort(v.begin(), v.end(),
            [](const ScoredMatch& a, const ScoredMatch& b) { return a.conf > b.conf; });
}

// ─── AP for one class from its confidence-sorted matches ───────────────────────

static float apForClass(const std::vector<ScoredMatch>& matches, int numGt) {
    if (numGt <= 0 || matches.empty()) return 0.0f;
    std::vector<float> recall, precision;
    recall.reserve(matches.size());
    precision.reserve(matches.size());
    int tp = 0, fp = 0;
    for (const ScoredMatch& m : matches) {
        if (m.tp) tp++; else fp++;
        recall.push_back((float)tp / numGt);
        precision.push_back((float)tp / (tp + fp));
    }
    return computeAP101(recall, precision);
}

// ─── Precision / recall / F1 sampled on NBINS confidence bins ──────────────────
// Bin b keeps predictions with conf >= b / NBINS, so the curves sweep the
// confidence axis from 0 to 1 (same as Ultralytics' np.interp(-x, -conf, ...)).

static void prfCurves(const std::vector<ScoredMatch>& matches, int numGt,
                      std::vector<float>& p, std::vector<float>& r,
                      std::vector<float>& f) {
    p.assign(NBINS, 1.0f);
    r.assign(NBINS, 0.0f);
    f.assign(NBINS, 0.0f);
    if (numGt <= 0) return;

    for (int b = 0; b < NBINS; b++) {
        float confCut = (float)b / NBINS;   // keep predictions with conf >= confCut
        int tp = 0, fp = 0;
        for (const ScoredMatch& m : matches) {
            if (m.conf < confCut) break;    // matches are sorted by conf desc
            if (m.tp) tp++; else fp++;
        }
        float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 1.0f;
        float rec  = (float)tp / numGt;
        p[b] = prec;
        r[b] = rec;
        f[b] = (prec + rec > 0.0f) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
    }
}

// Box filter with edge replication (Ultralytics' smooth(y, f)).
static std::vector<float> smooth(const std::vector<float>& y, float f) {
    int nf = (int)std::lround((float)y.size() * f * 2.0f);
    nf = (nf / 2) * 2 + 1;                      // force odd window
    if (nf < 1) nf = 1;
    if (nf >= (int)y.size()) return y;
    int half = nf / 2;

    std::vector<float> yp;
    yp.reserve(y.size() + 2 * half);
    yp.insert(yp.end(), half, y.front());
    yp.insert(yp.end(), y.begin(), y.end());
    yp.insert(yp.end(), half, y.back());

    double sum = 0.0;
    for (int i = 0; i < nf; i++) sum += yp[i];
    std::vector<float> out(y.size());
    for (size_t i = 0; i < y.size(); i++) {
        out[i] = (float)(sum / nf);
        sum += yp[i + nf] - yp[i];
    }
    return out;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════

bool runEvaluation(const EvalOptions& opt, float boxScale, int32_t boxOffset,
                   float scrScale, int32_t scrOffset) {
    // ── Resolve dataset directories ─────────────────────────────────────────
    std::string imgDir = opt.imagesDir;
    std::string lblDir = opt.labelsDir;
    if (!opt.dataDir.empty()) {
        imgDir = opt.dataDir + "/images";
        lblDir = opt.dataDir + "/labels";
    } else if (!imgDir.empty() && lblDir.empty()) {
        lblDir = parentDir(imgDir) + "/labels";   // standard sibling folder
    }

    std::string dsName = baseName(opt.dataDir.empty() ? parentDir(imgDir) : opt.dataDir);
    if (dsName.empty() || dsName == "/" || dsName == ".") dsName = "dataset";

    std::string outDir = opt.outDir;
    while (outDir.size() > 1 && outDir.back() == '/') outDir.pop_back();
    if (!dirExists(outDir)) {   // create the output directory if needed
        std::string mk = "mkdir -p " + shellQuote(outDir);
        if (system(mk.c_str()) != 0) {
            fprintf(stderr, "ERROR: cannot create output directory: %s\n", outDir.c_str());
            return false;
        }
    }
    std::string predCsv = outDir + "/predictions.csv";
    std::string sizeCsv = outDir + "/image_sizes.csv";
    std::string cfgCsv  = outDir + "/eval_config.csv";
    std::string imgList = outDir + "/evaluated_images.txt";
    std::string reportPath = opt.reportPath.empty()
                             ? outDir + "/eval_report_" + dsName + ".txt"
                             : opt.reportPath;

    // ── Error checks (12) ───────────────────────────────────────────────────
    if (imgDir.empty()) {
        fprintf(stderr, "ERROR: no image directory given (use --images or --map)\n");
        return false;
    }
    if (!dirExists(imgDir)) {
        fprintf(stderr, "ERROR: image directory does not exist: %s\n", imgDir.c_str());
        return false;
    }
    if (!dirExists(lblDir)) {
        fprintf(stderr, "ERROR: labels directory does not exist: %s\n", lblDir.c_str());
        return false;
    }

    std::vector<std::string> unsupported;
    std::vector<std::string> names = listImages(imgDir, unsupported);
    for (const std::string& u : unsupported)
        fprintf(stderr, "WARN: unsupported image type in %s — skipped: %s\n",
                imgDir.c_str(), u.c_str());
    if (names.empty()) {
        fprintf(stderr, "ERROR: no images (.jpg/.jpeg/.png/.bmp/.webp) found in %s\n",
                imgDir.c_str());
        return false;
    }

    // ── Model I/O validation against expected tensor dimensions ────────────
    uint32_t inDims[4] = {0,0,0,0}, mBoxDims[4] = {0,0,0,0}, mScrDims[4] = {0,0,0,0};
    uint32_t inRank = 0, boxRank = 0, scrRank = 0;
    if (!getModelShapes(inDims, inRank, mBoxDims, boxRank, mScrDims, scrRank)) {
        fprintf(stderr, "ERROR: model tensor information unavailable (load model first)\n");
        return false;
    }
    if (!(inDims[0] == 1 && inDims[1] == 3 && inDims[2] == 640 && inDims[3] == 640)) {
        fprintf(stderr, "ERROR: model input must be 1x3x640x640, got %s "
                        "(preprocessor is hard-wired to 640x640)\n",
                shapeStr(inDims, inRank).c_str());
        return false;
    }
    if (mBoxDims[1] != 4) {
        fprintf(stderr, "ERROR: boxes output must have shape 1x4xN, got %s\n",
                shapeStr(mBoxDims, boxRank).c_str());
        return false;
    }
    int modelNC  = (int)mScrDims[1];
    int anchors  = (int)mScrDims[2];
    if (modelNC <= 0 || anchors <= 0) {
        fprintf(stderr, "ERROR: unexpected scores output shape %s\n",
                shapeStr(mScrDims, scrRank).c_str());
        return false;
    }
    if (modelNC != opt.numClasses)
        fprintf(stderr, "WARN: model scores tensor has %d classes but --classes says %d "
                        "(using the model's %d)\n", modelNC, opt.numClasses, modelNC);
    if (anchors != 8400)
        fprintf(stderr, "WARN: expected 8400 anchors for 640x640 input, model has %d\n", anchors);

    // ── Reproducibility header (13) ─────────────────────────────────────────
    printf("\n============================================================\n"
           "  QNN HTP NPU STATIC DETECTION EVALUATION\n"
           "============================================================\n");
    printf("  Runtime/backend    : %s\n", opt.runtimeName.c_str());
    printf("  Model (context bin): %s\n", opt.modelPath.c_str());
    printf("  Images directory   : %s   (%zu images)\n", imgDir.c_str(), names.size());
    printf("  Labels directory   : %s\n", lblDir.c_str());
    printf("  Input resolution   : %ux%u  (tensor %s)\n",
           inDims[2], inDims[3], shapeStr(inDims, inRank).c_str());
    printf("  Output tensors     : boxes %s, scores %s\n",
           shapeStr(mBoxDims, boxRank).c_str(), shapeStr(mScrDims, scrRank).c_str());
    printf("  Classes            : %d\n", modelNC);
    printf("  Conf retention     : %.3f\n", opt.confThresh);
    printf("  NMS IoU            : %.2f\n", opt.iouThresh);
    printf("  Max detections     : %d\n", opt.maxDet);
    printf("  Warm-up images     : %d (excluded from timing, kept in accuracy)\n",
           opt.warmupFrames);
    printf("  Output directory   : %s\n", outDir.c_str());
    printf("------------------------------------------------------------\n");

    // ── Open output files ───────────────────────────────────────────────────
    std::ofstream pcsv(predCsv);
    if (!pcsv.is_open()) {
        fprintf(stderr, "ERROR: cannot write %s\n", predCsv.c_str());
        return false;
    }
    pcsv << std::setprecision(9);   // full float precision, no rounding
    pcsv << "image,class_id,confidence,x1,y1,x2,y2\n";

    std::ofstream scsv(sizeCsv);
    if (!scsv.is_open()) {
        fprintf(stderr, "ERROR: cannot write %s\n", sizeCsv.c_str());
        return false;
    }
    scsv << "image,width,height\n";

    std::ofstream ilst(imgList);
    if (!ilst.is_open()) {
        fprintf(stderr, "ERROR: cannot write %s\n", imgList.c_str());
        return false;
    }

    // ── Run the pipeline on every image ─────────────────────────────────────
    // The first `warmup` images are processed normally (their predictions DO
    // count for accuracy) but their timings are excluded from the statistics.
    int warmup = std::max(0, std::min(opt.warmupFrames, (int)names.size() - 1));
    std::vector<EvalImage> images;
    images.reserve(names.size());

    int totalGt = 0, totalPreds = 0, done = 0, skipped = 0, missingLabels = 0;
    std::vector<double> preTimes, infTimes, postTimes, totTimes;
    auto startAll = Clock::now();

    for (size_t ni = 0; ni < names.size(); ni++) {
        const std::string& name = names[ni];
        std::string imgPath = imgDir + "/" + name;

        cv::Mat raw = cv::imread(imgPath);
        if (raw.empty()) {
            fprintf(stderr, "WARN: cannot read image %s — skipped\n", imgPath.c_str());
            skipped++;
            continue;
        }

        EvalImage img;
        img.name = name;
        img.w = raw.cols;
        img.h = raw.rows;

        // ── Ground truth (YOLO txt: "cls cx cy w h", normalized) ───────────
        std::string lblPath = lblDir + "/" + stemOf(name) + ".txt";
        std::ifstream lf(lblPath);
        if (!lf.is_open()) {
            // Valid empty-image case: no label file = no objects.
            missingLabels++;
        } else {
            std::string line;
            int lineno = 0;
            while (std::getline(lf, line)) {
                lineno++;
                line = trim(line);
                if (line.empty()) continue;   // empty line = no box

                int cls = -1;
                float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f;
                char extra = 0;
                int n = sscanf(line.c_str(), "%d %f %f %f %f %c",
                               &cls, &cx, &cy, &w, &h, &extra);
                if (n < 5) {
                    fprintf(stderr, "WARN: malformed label %s:%d (%s) — ignored\n",
                            lblPath.c_str(), lineno, line.c_str());
                    continue;
                }
                if (cls < 0 || cls >= modelNC) {
                    fprintf(stderr, "WARN: class id %d out of range [0,%d) in %s:%d — ignored\n",
                            cls, modelNC, lblPath.c_str(), lineno);
                    continue;
                }
                if (!(w > 0.0f) || !(h > 0.0f)) {
                    fprintf(stderr, "WARN: non-positive box size in %s:%d — ignored\n",
                            lblPath.c_str(), lineno);
                    continue;
                }
                // Convert to original-image pixel coordinates (no clipping —
                // Ultralytics val.py does not clip either; IoU matching
                // handles boxes that slightly exceed the image bounds).
                float x1 = (cx - w * 0.5f) * img.w;
                float y1 = (cy - h * 0.5f) * img.h;
                float x2 = (cx + w * 0.5f) * img.w;
                float y2 = (cy + h * 0.5f) * img.h;
                img.gt.push_back({ cls, x1, y1, x2, y2 });
            }
        }
        totalGt += (int)img.gt.size();

        // ── Preprocess ──────────────────────────────────────────────────────
        auto t0 = Clock::now();
        PreprocessResult pre = preprocess(raw);
        auto t1 = Clock::now();
        if (pre.data.empty()) { skipped++; continue; }

        // ── QNN/NPU inference ───────────────────────────────────────────────
        uint8_t* boxBuf = nullptr;
        uint8_t* scrBuf = nullptr;
        uint32_t boxDims[4] = { 0, 0, 0, 0 }, scrDims[4] = { 0, 0, 0, 0 };
        if (!runInference(pre.data, boxBuf, boxDims, scrBuf, scrDims)) {
            fprintf(stderr, "ERROR: inference failed on %s\n", name.c_str());
            return false;
        }
        auto t2 = Clock::now();

        // ── Postprocess: dequant + decode + NMS + max-det + reverse letterbox ──
        std::vector<Detection> dets = parseOutputs(boxBuf, scrBuf, boxDims, scrDims,
                                                   boxScale, boxOffset,
                                                   scrScale, scrOffset,
                                                   opt.confThresh);
        std::vector<Detection> nd = nms(dets, opt.confThresh, opt.iouThresh);
        auto t3 = Clock::now();
        free(boxBuf);
        free(scrBuf);

        // Cap at maxDet (COCO/Ultralytics convention): keep highest confidence.
        if ((int)nd.size() > opt.maxDet) {
            std::stable_sort(nd.begin(), nd.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence > b.confidence;
                });
            nd.resize((size_t)opt.maxDet);
        }

        // ── Reverse letterbox: 640-space → original pixel coordinates ───────
        // Exact inverse of preprocess: the resized content occupies
        // [padLeft, padLeft+resizedW) x [padTop, padTop+resizedH) inside the
        // 640x640 tensor, resized by factor resizedW/origW.  Using the exact
        // resize dimensions avoids the sub-pixel error of dividing by the
        // un-rounded scale.  Boxes are NOT clipped to the image bounds —
        // Ultralytics val.py uses unclipped boxes as well.
        double invX = pre.resizedW > 0 ? (double)img.w / pre.resizedW
                                       : 1.0 / (double)pre.scale;
        double invY = pre.resizedH > 0 ? (double)img.h / pre.resizedH
                                       : 1.0 / (double)pre.scale;
        for (Detection& d : nd) {
            d.x1 = (float)((d.x1 - pre.padLeft) * invX);
            d.y1 = (float)((d.y1 - pre.padTop)  * invY);
            d.x2 = (float)((d.x2 - pre.padLeft) * invX);
            d.y2 = (float)((d.y2 - pre.padTop)  * invY);
        }
        img.preds = std::move(nd);
        totalPreds += (int)img.preds.size();

        // ── Save raw detections for audit / evaluate_metrics.py (5) ─────────
        for (const Detection& d : img.preds)
            pcsv << name << ',' << d.classId << ',' << d.confidence << ','
                 << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << '\n';
        scsv << name << ',' << img.w << ',' << img.h << '\n';
        ilst << name << '\n';

        // ── Timing (warm-up excluded) (8, 9) ────────────────────────────────
        double preMs  = Ms(t1 - t0).count();
        double infMs  = Ms(t2 - t1).count();
        double postMs = Ms(t3 - t2).count();
        bool timed = (int)ni >= warmup;
        if (timed) {
            preTimes.push_back(preMs);
            infTimes.push_back(infMs);
            postTimes.push_back(postMs);
            totTimes.push_back(preMs + infMs + postMs);
        }
        done++;

        printf("  [%3d/%zu] %-52s gt:%2d det:%3d  %.1f/%.1f/%.1f ms%s\n",
               done, names.size(), name.c_str(),
               (int)img.gt.size(), (int)img.preds.size(),
               preMs, infMs, postMs, timed ? "" : "  (warm-up)");

        images.push_back(std::move(img));
    }

    pcsv.close();
    scsv.close();
    ilst.close();
    double wallSec = Ms(Clock::now() - startAll).count() / 1000.0;

    if (done == 0) {
        fprintf(stderr, "ERROR: no images were processed\n");
        return false;
    }

    // ── Per-class GT statistics ──────────────────────────────────────────────
    std::vector<int> gtCount((size_t)modelNC, 0);
    std::vector<int> imgCount((size_t)modelNC, 0);
    for (const EvalImage& img : images) {
        for (int c = 0; c < modelNC; c++) {
            bool seen = false;
            for (const GtBox& g : img.gt)
                if (g.cls == c) { gtCount[c]++; seen = true; }
            if (seen) imgCount[c]++;
        }
    }

    // ── AP per class per IoU threshold ───────────────────────────────────────
    std::vector<std::vector<float>> ap((size_t)NUM_IOU_THRESH,
                                       std::vector<float>((size_t)modelNC, 0.0f));
    for (int t = 0; t < NUM_IOU_THRESH; t++) {
        float thr = 0.5f + 0.05f * t;
        std::vector<std::vector<ScoredMatch>> perClass;
        gatherMatches(images, thr, modelNC, perClass);
        for (int c = 0; c < modelNC; c++)
            if (gtCount[c] > 0)
                ap[t][c] = apForClass(perClass[c], gtCount[c]);
    }

    // Overall mAP: mean over classes that appear in GT (COCO rule)
    std::vector<float> mapAtIou((size_t)NUM_IOU_THRESH, 0.0f);
    std::vector<float> ap50c((size_t)modelNC, 0.0f);
    std::vector<float> ap5095c((size_t)modelNC, 0.0f);
    int validClasses = 0;
    for (int c = 0; c < modelNC; c++) {
        if (gtCount[c] == 0) continue;
        validClasses++;
        ap50c[c] = ap[0][c];
        float acc = 0.0f;
        for (int t = 0; t < NUM_IOU_THRESH; t++) {
            acc += ap[t][c];
            mapAtIou[t] += ap[t][c];
        }
        ap5095c[c] = acc / NUM_IOU_THRESH;
    }
    float map50 = 0.0f, map5095 = 0.0f;
    if (validClasses > 0) {
        for (int t = 0; t < NUM_IOU_THRESH; t++) mapAtIou[t] /= validClasses;
        map50 = mapAtIou[0];
        for (float v : mapAtIou) map5095 += v;
        map5095 /= NUM_IOU_THRESH;
    }

    // ── P / R / F1 at IoU 0.5, max-F1 operating point (C++ built-in) ─────────
    std::vector<float> pc((size_t)modelNC, 0.0f);
    std::vector<float> rc((size_t)modelNC, 0.0f);
    std::vector<float> fc((size_t)modelNC, 0.0f);
    float bestConf = 0.0f;
    {
        std::vector<std::vector<ScoredMatch>> perClass;
        gatherMatches(images, 0.5f, modelNC, perClass);

        std::vector<std::vector<float>> pC((size_t)modelNC);
        std::vector<std::vector<float>> rC((size_t)modelNC);
        std::vector<std::vector<float>> fC((size_t)modelNC);
        for (int c = 0; c < modelNC; c++)
            if (gtCount[c] > 0)
                prfCurves(perClass[c], gtCount[c], pC[c], rC[c], fC[c]);

        if (validClasses > 0) {
            // Smoothed mean-F1 curve; argmax = operating point
            std::vector<float> meanF(NBINS, 0.0f);
            for (int c = 0; c < modelNC; c++) {
                if (gtCount[c] == 0) continue;
                for (int b = 0; b < NBINS; b++) meanF[b] += fC[c][b];
            }
            for (float& v : meanF) v /= validClasses;
            std::vector<float> smF = smooth(meanF, 0.1f);
            int best = (int)(std::max_element(smF.begin(), smF.end()) - smF.begin());
            bestConf = (float)best / NBINS;

            for (int c = 0; c < modelNC; c++) {
                if (gtCount[c] == 0) continue;
                pc[c] = pC[c][best];
                rc[c] = rC[c][best];
                fc[c] = fC[c][best];
            }
        }
    }
    float P = 0.0f, R = 0.0f, F1 = 0.0f;
    if (validClasses > 0) {
        for (int c = 0; c < modelNC; c++) {
            if (gtCount[c] == 0) continue;
            P += pc[c]; R += rc[c]; F1 += fc[c];
        }
        P /= validClasses;
        R /= validClasses;
        F1 /= validClasses;
    }

    // ── Timing statistics (8) ────────────────────────────────────────────────
    TimeStats sPre = timeStats(preTimes), sInf = timeStats(infTimes),
              sPost = timeStats(postTimes), sTot = timeStats(totTimes);
    int timedN = (int)preTimes.size();

    char dateBuf[32] = {};
    {
        time_t now = time(nullptr);
        strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    }

    // ── eval_config.csv (audit metadata for evaluate_metrics.py) ────────────
    {
        std::ofstream cf(cfgCsv);
        if (cf.is_open()) {
            cf << "key,value\n";
            auto put = [&cf](const std::string& k, const std::string& v) {
                cf << k << ",\"" << v << "\"\n";
            };
            put("model/runtime", opt.runtimeName);
            put("model_path", opt.modelPath);
            put("runtime/backend", opt.runtimeName);
            put("input_resolution", fmt("%ux%u", inDims[2], inDims[3]));
            put("num_classes", fmt("%d", modelNC));
            put("num_images", fmt("%d", done));
            put("ground_truth_boxes", fmt("%d", totalGt));
            put("predicted_boxes", fmt("%d", totalPreds));
            put("confidence_retention_threshold", fmt("%.4f", opt.confThresh));
            put("nms_iou", fmt("%.4f", opt.iouThresh));
            put("max_det", fmt("%d", opt.maxDet));
            put("warmup_frames", fmt("%d", warmup));
            put("images_dir", imgDir);
            put("labels_dir", lblDir);
            put("date", dateBuf);
            cf.close();
        } else {
            fprintf(stderr, "WARN: cannot write %s\n", cfgCsv.c_str());
        }
    }

    // ── Build report ─────────────────────────────────────────────────────────
    double imgPerS = wallSec > 0.0 ? done / wallSec : 0.0;

    std::string rep;
    auto add = [&rep](const std::string& s) { rep += s; rep += '\n'; };

    add("============================================================");
    add("  YOLOv8 Evaluation Report (COCO-style mAP)");
    add("============================================================");
    add(fmt("  Date:            %s", dateBuf));
    add("  Runtime/backend: " + opt.runtimeName);
    add("  Dataset:         " + imgDir);
    add("  Labels:          " + lblDir);
    add("  Model:           " + opt.modelPath);
    add(fmt("  Input:           %s", shapeStr(inDims, inRank).c_str()));
    add(fmt("  Outputs:         boxes %s, scores %s",
            shapeStr(mBoxDims, boxRank).c_str(),
            shapeStr(mScrDims, scrRank).c_str()));
    add(fmt("  Images:          %d", done));
    add(fmt("  GT boxes:        %d", totalGt));
    add(fmt("  Predictions:     %d", totalPreds));
    add(fmt("  Conf: %.3f   NMS IoU: %.2f   Classes: %d   Max det: %d",
            opt.confThresh, opt.iouThresh, modelNC, opt.maxDet));
    add(fmt("  Warm-up images:  %d (excluded from timing, kept in accuracy)", warmup));
    add("------------------------------------------------------------");
    add(fmt("  mAP@0.5          %.4f", map50));
    add(fmt("  mAP@0.5:0.95     %.4f", map5095));
    add(fmt("  Precision        %.4f", P));
    add(fmt("  Recall           %.4f", R));
    add(fmt("  F1-score         %.4f", F1));
    add(fmt("  Best conf        %.3f   (operating point of P/R/F1)", bestConf));
    add("------------------------------------------------------------");
    add("  AP per IoU threshold:");
    add("    IoU      AP");
    for (int t = 0; t < NUM_IOU_THRESH; t++)
        add(fmt("    %.2f    %.4f", 0.5f + 0.05f * t, mapAtIou[t]));
    add("------------------------------------------------------------");
    add("  Per class:");
    add("    Class              Img    GT    AP50    AP50-95   Prec    Rec     F1");
    for (int c = 0; c < modelNC; c++) {
        if (gtCount[c] == 0) continue;
        add(fmt("    %-16s %4d %5d   %6.4f   %6.4f   %6.4f  %6.4f  %6.4f",
                className(c), imgCount[c], gtCount[c],
                ap50c[c], ap5095c[c], pc[c], rc[c], fc[c]));
    }
    add("------------------------------------------------------------");
    add("  Timing (warm-up excluded; std = population stddev):");
    add(fmt("    Images timed:      %d", timedN));
    add(fmt("    Total:             %.3f s   (%.1f images/s, %d images)",
            wallSec, imgPerS, done));
    add("                        mean       std       min       max");
    add(fmt("    preprocess  (ms) %8.3f  %8.3f  %8.3f  %8.3f",
            sPre.mean, sPre.sd, sPre.mn, sPre.mx));
    add(fmt("    QNN inference(ms)%8.3f  %8.3f  %8.3f  %8.3f",
            sInf.mean, sInf.sd, sInf.mn, sInf.mx));
    add(fmt("    postprocess (ms) %8.3f  %8.3f  %8.3f  %8.3f",
            sPost.mean, sPost.sd, sPost.mn, sPost.mx));
    add(fmt("    total       (ms) %8.3f  %8.3f  %8.3f  %8.3f",
            sTot.mean, sTot.sd, sTot.mn, sTot.mx));
    if (skipped > 0)
        add(fmt("  Skipped (unreadable): %d", skipped));
    if (missingLabels > 0)
        add(fmt("  Images without label file (treated as empty): %d", missingLabels));
    add("------------------------------------------------------------");
    add(fmt("  Files: %s", predCsv.c_str()));
    add(fmt("         %s", sizeCsv.c_str()));
    add(fmt("         %s", cfgCsv.c_str()));
    add(fmt("         %s", imgList.c_str()));
    add("------------------------------------------------------------");
    add("  Evaluated image files:");
    for (const EvalImage& img : images)
        add("    " + img.name);
    add("============================================================");

    // ── Write report ────────────────────────────────────────────────────────
    std::ofstream of(reportPath);
    if (of.is_open()) {
        of << rep;
        of.close();
    } else {
        fprintf(stderr, "WARN: cannot write report to %s\n", reportPath.c_str());
    }

    // ── Final console output (7) ────────────────────────────────────────────
    printf("\n============================================================\n");
    printf("QNN HTP NPU STATIC DETECTION EVALUATION\n");
    printf("============================================================\n");
    printf("Images              : %d\n", done);
    printf("Ground-truth boxes  : %d\n", totalGt);
    printf("Predicted boxes     : %d\n", totalPreds);
    printf("\n");
    printf("mAP@0.5              : %.4f\n", map50);
    printf("mAP@0.5:0.95         : %.4f\n", map5095);
    printf("Precision            : %.4f\n", P);
    printf("Recall               : %.4f\n", R);
    printf("F1                   : %.4f\n", F1);
    printf("\n");
    printf("Prediction threshold : %.3f\n", opt.confThresh);
    printf("NMS IoU              : %.2f\n", opt.iouThresh);
    printf("Max detections       : %d\n", opt.maxDet);
    printf("\n");
    printf("(C++ built-in COCO 101-point metrics above; the canonical\n"
           " Ultralytics-exact numbers are printed by evaluate_metrics.py)\n");
    printf("\n");
    printf("--- Timing (%d warm-up excluded, %d timed images) ---\n", warmup, timedN);
    printf("                       mean       std       min       max\n");
    printf("  preprocess  (ms) %8.3f  %8.3f  %8.3f  %8.3f\n",
           sPre.mean, sPre.sd, sPre.mn, sPre.mx);
    printf("  QNN inference(ms)%8.3f  %8.3f  %8.3f  %8.3f\n",
           sInf.mean, sInf.sd, sInf.mn, sInf.mx);
    printf("  postprocess (ms) %8.3f  %8.3f  %8.3f  %8.3f\n",
           sPost.mean, sPost.sd, sPost.mn, sPost.mx);
    printf("  total       (ms) %8.3f  %8.3f  %8.3f  %8.3f\n",
           sTot.mean, sTot.sd, sTot.mn, sTot.mx);
    printf("\n");
    printf("  Total images      : %d\n", done);
    printf("  Total processing  : %.3f s\n", wallSec);
    printf("  Images/second     : %.2f\n", imgPerS);
    if (skipped > 0)      printf("  Skipped images    : %d (unreadable)\n", skipped);
    if (missingLabels > 0) printf("  No label file     : %d images (treated as empty)\n", missingLabels);
    printf("============================================================\n");

    printf("\nSaved:\n");
    printf("  %s\n  %s\n  %s\n  %s\n  %s\n",
           predCsv.c_str(), sizeCsv.c_str(), cfgCsv.c_str(),
           imgList.c_str(), reportPath.c_str());

    // ── Ultralytics-exact metrics via evaluate_metrics.py (6) ───────────────
    fflush(stdout);   // keep console output ordered before the subprocess runs
    runPythonMetrics(opt, predCsv, lblDir, cfgCsv, outDir + "/evaluation_results.csv");

    return true;
}
