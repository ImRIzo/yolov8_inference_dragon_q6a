/**
 * evaluate.cpp — COCO-style mAP evaluation (mAP@0.5, mAP@0.5:0.95, P/R/F1)
 *
 * Walks a dataset folder (images/ + labels/ in YOLO txt format), runs the
 * full QNN pipeline on every image and computes:
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
#include <string>
#include <vector>

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

// ── Small helpers ─────────────────────────────────────────────────────────────

static std::string fmt(const char* format, ...) {
    char buf[512];
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

static std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        for (const char* ext : IMG_EXTS)
            if (hasExt(name, ext)) { out.push_back(name); break; }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// ── IoU ───────────────────────────────────────────────────────────────────────

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

// ── 101-point interpolated AP (equivalent of np.trapz(np.interp(x, mrec, mpre))) ──

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

// ── Greedy per-image matching ─────────────────────────────────────────────────
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

// ── AP for one class from its confidence-sorted matches ───────────────────────

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

// ── Precision / recall / F1 sampled on NBINS confidence bins ──────────────────
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
                   float scrScale, int32_t scrOffset,
                   const std::string& modelPath) {
    const std::string imgDir = opt.dataDir + "/images";
    const std::string lblDir = opt.dataDir + "/labels";

    std::vector<std::string> names = listImages(imgDir);
    if (names.empty()) {
        fprintf(stderr, "ERROR: no images found in %s\n", imgDir.c_str());
        return false;
    }

    printf("\n============================================================\n"
           "  YOLOv8 Evaluation — COCO-style mAP\n"
           "============================================================\n");
    printf("  Dataset:   %s\n", opt.dataDir.c_str());
    printf("  Images:    %zu\n", names.size());
    printf("  Model:     %s\n", modelPath.c_str());
    printf("  Conf: %.3f   NMS IoU: %.2f   Classes: %d   Max det: %d\n",
           opt.confThresh, opt.iouThresh, opt.numClasses, opt.maxDet);
    printf("------------------------------------------------------------\n");

    // ── Run the pipeline on every image ─────────────────────────────────────
    std::vector<EvalImage> images;
    images.reserve(names.size());

    int totalGt = 0, totalPreds = 0, done = 0;
    double totalPreMs = 0.0, totalInfMs = 0.0, totalPostMs = 0.0;
    auto startAll = Clock::now();

    for (const std::string& name : names) {
        std::string imgPath = imgDir + "/" + name;
        cv::Mat raw = cv::imread(imgPath);
        if (raw.empty()) {
            fprintf(stderr, "WARN: cannot read %s — skipped\n", imgPath.c_str());
            continue;
        }

        EvalImage img;
        img.name = name;
        img.w = raw.cols;
        img.h = raw.rows;

        // Ground truth (YOLO txt: "cls cx cy w h", normalized to image size)
        std::string lblPath = lblDir + "/" + name.substr(0, name.rfind('.')) + ".txt";
        std::ifstream lf(lblPath);
        if (!lf.is_open()) {
            fprintf(stderr, "WARN: no label file %s — treated as empty\n", lblPath.c_str());
        } else {
            std::string line;
            while (std::getline(lf, line)) {
                int cls = -1;
                float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f;
                if (sscanf(line.c_str(), "%d %f %f %f %f", &cls, &cx, &cy, &w, &h) != 5)
                    continue;
                if (cls < 0 || cls >= opt.numClasses || w <= 0.0f || h <= 0.0f) continue;
                float x1 = (cx - w * 0.5f) * img.w;
                float y1 = (cy - h * 0.5f) * img.h;
                float x2 = (cx + w * 0.5f) * img.w;
                float y2 = (cy + h * 0.5f) * img.h;
                x1 = std::max(0.0f, x1); y1 = std::max(0.0f, y1);
                x2 = std::min((float)img.w, x2); y2 = std::min((float)img.h, y2);
                img.gt.push_back({ cls, x1, y1, x2, y2 });
            }
        }
        totalGt += (int)img.gt.size();

        // Preprocess
        auto t0 = Clock::now();
        PreprocessResult pre = preprocess(raw);
        auto t1 = Clock::now();
        if (pre.data.empty()) continue;

        // Inference
        uint8_t* boxBuf = nullptr;
        uint8_t* scrBuf = nullptr;
        uint32_t boxDims[4] = { 0, 0, 0, 0 }, scrDims[4] = { 0, 0, 0, 0 };
        if (!runInference(pre.data, boxBuf, boxDims, scrBuf, scrDims)) {
            fprintf(stderr, "ERROR: inference failed on %s\n", name.c_str());
            return false;
        }
        auto t2 = Clock::now();

        // Postprocess
        std::vector<Detection> dets = parseOutputs(boxBuf, scrBuf, boxDims, scrDims,
                                                   boxScale, boxOffset,
                                                   scrScale, scrOffset,
                                                   opt.confThresh);
        std::vector<Detection> nd = nms(dets, opt.confThresh, opt.iouThresh);
        auto t3 = Clock::now();
        free(boxBuf);
        free(scrBuf);

        // Cap at maxDet (COCO convention)
        if ((int)nd.size() > opt.maxDet) {
            std::stable_sort(nd.begin(), nd.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence > b.confidence;
                });
            nd.resize((size_t)opt.maxDet);
        }

        // Back-project from padded 640-space to original pixel coordinates
        for (Detection& d : nd) {
            float x1 = (d.x1 - pre.padLeft) / pre.scale;
            float y1 = (d.y1 - pre.padTop) / pre.scale;
            float x2 = (d.x2 - pre.padLeft) / pre.scale;
            float y2 = (d.y2 - pre.padTop) / pre.scale;
            d.x1 = std::max(0.0f, std::min((float)img.w, x1));
            d.y1 = std::max(0.0f, std::min((float)img.h, y1));
            d.x2 = std::max(0.0f, std::min((float)img.w, x2));
            d.y2 = std::max(0.0f, std::min((float)img.h, y2));
        }
        img.preds = std::move(nd);
        totalPreds += (int)img.preds.size();

        double preMs  = Ms(t1 - t0).count();
        double infMs  = Ms(t2 - t1).count();
        double postMs = Ms(t3 - t2).count();
        totalPreMs  += preMs;
        totalInfMs  += infMs;
        totalPostMs += postMs;
        done++;

        printf("  [%3d/%zu] %-52s gt:%2d det:%3d  %.1f/%.1f/%.1f ms\n",
               done, names.size(), name.c_str(),
               (int)img.gt.size(), (int)img.preds.size(),
               preMs, infMs, postMs);

        images.push_back(std::move(img));
    }

    if (images.empty()) {
        fprintf(stderr, "ERROR: no images were processed\n");
        return false;
    }

    // ── Per-class GT statistics ──────────────────────────────────────────────
    std::vector<int> gtCount((size_t)opt.numClasses, 0);
    std::vector<int> imgCount((size_t)opt.numClasses, 0);
    for (const EvalImage& img : images) {
        for (int c = 0; c < opt.numClasses; c++) {
            bool seen = false;
            for (const GtBox& g : img.gt)
                if (g.cls == c) { gtCount[c]++; seen = true; }
            if (seen) imgCount[c]++;
        }
    }

    // ── AP per class per IoU threshold ───────────────────────────────────────
    std::vector<std::vector<float>> ap((size_t)NUM_IOU_THRESH,
                                       std::vector<float>((size_t)opt.numClasses, 0.0f));
    for (int t = 0; t < NUM_IOU_THRESH; t++) {
        float thr = 0.5f + 0.05f * t;
        std::vector<std::vector<ScoredMatch>> perClass;
        gatherMatches(images, thr, opt.numClasses, perClass);
        for (int c = 0; c < opt.numClasses; c++)
            if (gtCount[c] > 0)
                ap[t][c] = apForClass(perClass[c], gtCount[c]);
    }

    // Overall mAP: mean over classes that appear in GT (COCO rule)
    std::vector<float> mapAtIou((size_t)NUM_IOU_THRESH, 0.0f);
    std::vector<float> ap50c((size_t)opt.numClasses, 0.0f);
    std::vector<float> ap5095c((size_t)opt.numClasses, 0.0f);
    int validClasses = 0;
    for (int c = 0; c < opt.numClasses; c++) {
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

    // ── P / R / F1 at IoU 0.5, max-F1 operating point ────────────────────────
    std::vector<float> pc((size_t)opt.numClasses, 0.0f);
    std::vector<float> rc((size_t)opt.numClasses, 0.0f);
    std::vector<float> fc((size_t)opt.numClasses, 0.0f);
    float bestConf = 0.0f;
    {
        std::vector<std::vector<ScoredMatch>> perClass;
        gatherMatches(images, 0.5f, opt.numClasses, perClass);

        std::vector<std::vector<float>> pC((size_t)opt.numClasses);
        std::vector<std::vector<float>> rC((size_t)opt.numClasses);
        std::vector<std::vector<float>> fC((size_t)opt.numClasses);
        for (int c = 0; c < opt.numClasses; c++)
            if (gtCount[c] > 0)
                prfCurves(perClass[c], gtCount[c], pC[c], rC[c], fC[c]);

        if (validClasses > 0) {
            // Smoothed mean-F1 curve; argmax = operating point
            std::vector<float> meanF(NBINS, 0.0f);
            for (int c = 0; c < opt.numClasses; c++) {
                if (gtCount[c] == 0) continue;
                for (int b = 0; b < NBINS; b++) meanF[b] += fC[c][b];
            }
            for (float& v : meanF) v /= validClasses;
            std::vector<float> smF = smooth(meanF, 0.1f);
            int best = (int)(std::max_element(smF.begin(), smF.end()) - smF.begin());
            bestConf = (float)best / NBINS;

            for (int c = 0; c < opt.numClasses; c++) {
                if (gtCount[c] == 0) continue;
                pc[c] = pC[c][best];
                rc[c] = rC[c][best];
                fc[c] = fC[c][best];
            }
        }
    }
    float P = 0.0f, R = 0.0f, F1 = 0.0f;
    if (validClasses > 0) {
        for (int c = 0; c < opt.numClasses; c++) {
            if (gtCount[c] == 0) continue;
            P += pc[c]; R += rc[c]; F1 += fc[c];
        }
        P /= validClasses;
        R /= validClasses;
        F1 /= validClasses;
    }

    // ── Build report ─────────────────────────────────────────────────────────
    double totalSec = Ms(Clock::now() - startAll).count() / 1000.0;
    double avgPre  = done ? totalPreMs / done : 0.0;
    double avgInf  = done ? totalInfMs / done : 0.0;
    double avgPost = done ? totalPostMs / done : 0.0;
    double imgPerS = totalSec > 0.0 ? done / totalSec : 0.0;

    char dateBuf[32] = {};
    {
        time_t now = time(nullptr);
        strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    }

    std::string rep;
    auto add = [&rep](const std::string& s) { rep += s; rep += '\n'; };

    add("============================================================");
    add("  YOLOv8 Evaluation Report (COCO-style mAP)");
    add("============================================================");
    add(fmt("  Date:            %s", dateBuf));
    add("  Dataset:         " + opt.dataDir);
    add("  Model:           " + modelPath);
    add(fmt("  Images:          %d", done));
    add(fmt("  GT boxes:        %d", totalGt));
    add(fmt("  Predictions:     %d", totalPreds));
    add(fmt("  Conf: %.3f   NMS IoU: %.2f   Classes: %d   Max det: %d",
            opt.confThresh, opt.iouThresh, opt.numClasses, opt.maxDet));
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
    for (int c = 0; c < opt.numClasses; c++) {
        if (gtCount[c] == 0) continue;
        add(fmt("    %-16s %4d %5d   %6.4f   %6.4f   %6.4f  %6.4f  %6.4f",
                className(c), imgCount[c], gtCount[c],
                ap50c[c], ap5095c[c], pc[c], rc[c], fc[c]));
    }
    add("------------------------------------------------------------");
    add("  Timing:");
    add(fmt("    Total:           %.1f s   (%.1f images/s)", totalSec, imgPerS));
    add(fmt("    Avg preprocess:  %.1f ms", avgPre));
    add(fmt("    Avg inference:   %.1f ms", avgInf));
    add(fmt("    Avg postprocess: %.1f ms", avgPost));
    add(fmt("    Avg total:       %.1f ms", avgPre + avgInf + avgPost));
    add("============================================================");

    // ── Write to file ────────────────────────────────────────────────────────
    std::ofstream of(opt.reportPath);
    if (of.is_open()) {
        of << rep;
        of.close();
    } else {
        fprintf(stderr, "WARN: cannot write report to %s\n", opt.reportPath.c_str());
    }

    printf("%s", rep.c_str());
    printf("\nReport saved: %s\n", opt.reportPath.c_str());
    return true;
}
