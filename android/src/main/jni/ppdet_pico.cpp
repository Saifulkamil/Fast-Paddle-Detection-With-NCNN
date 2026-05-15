// PicoDet NCNN Detection Engine - Implementation
//
// The shipped model bakes post-processing into the graph (NonMaxSuppression,
// TopK, Gather, F.embedding, Tensor.to, pnnx.Expression). ncnn does not have
// any of those ops natively, so we register no-op stubs for them and extract
// upstream blobs that only depend on standard ncnn layers.
//
// Intermediate blobs we read:
//   "317" : class scores      [w = num_anchors, h = num_class]
//   "339" : decoded boxes xyxy [w = 4,           h = num_anchors]
//           (already in 320x320 input pixel coords)
//
// We then run threshold + class-aware NMS, and inverse the letterbox.

#include "ppdet_pico.h"

#include "cpu.h"
#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <android/log.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#define TAG "PicoDet"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ============================================================================
// No-op stub layer used to satisfy load_param for unknown ops.
// ncnn evaluates lazily, so as long as we never extract a downstream blob,
// these forwards are never called. We still implement them defensively.
// ============================================================================

namespace {

class NoopLayer : public ncnn::Layer
{
public:
    NoopLayer()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& /*pd*/) override
    {
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& /*bottom_blobs*/,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const override
    {
        // Produce empty mats so downstream code that does extract on these
        // (we never do, but defensively) doesn't dereference null.
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            top_blobs[i] = ncnn::Mat(1, (size_t)4u, opt.blob_allocator);
        }
        return 0;
    }
};

::ncnn::Layer* noop_layer_creator(void* /*userdata*/)
{
    return new NoopLayer;
}

// ----- helpers for NMS -----

inline float intersection_area(const DetObject& a, const DetObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void qsort_descent(std::vector<DetObject>& objs, int left, int right)
{
    int i = left;
    int j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j)
    {
        while (objs[i].prob > p) i++;
        while (objs[j].prob < p) j--;
        if (i <= j)
        {
            std::swap(objs[i], objs[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent(objs, left, j);
    if (i < right) qsort_descent(objs, i, right);
}

void qsort_descent(std::vector<DetObject>& objs)
{
    if (objs.empty()) return;
    qsort_descent(objs, 0, (int)objs.size() - 1);
}

void nms_class_aware(const std::vector<DetObject>& objs,
                     std::vector<int>& picked,
                     float nms_threshold)
{
    picked.clear();
    const int n = (int)objs.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = objs[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const DetObject& a = objs[i];
        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const DetObject& b = objs[picked[j]];
            if (a.label != b.label) continue;
            float inter = intersection_area(a, b);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (uni > 0 && inter / uni > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

// Read a value from a 2D ncnn::Mat treated as [h rows, w cols], with row()
// returning the pointer to row r.
inline float at2d(const ncnn::Mat& m, int r, int c)
{
    return m.row(r)[c];
}

} // namespace

// ============================================================================
// PicoDet
// ============================================================================

PicoDet::PicoDet()
    : target_size(320),
      num_class(1),
      prob_threshold(0.4f),
      nms_threshold(0.5f),
      anti_spoof_enabled(false)
{
}

PicoDet::~PicoDet()
{
    picodet_net.clear();
}

static void register_stubs(ncnn::Net& net)
{
    // Each unknown type gets the same noop creator. They appear only in the
    // post-processing tail of the graph and we never extract any blob beyond
    // the intermediate ones, so they are never invoked.
    const char* unknown_types[] = {
        "pnnx.Expression",
        "NonMaxSuppression",
        "TopK",
        "Gather",
        "F.embedding",
        "Tensor.to",
    };
    for (const char* t : unknown_types)
    {
        int ret = net.register_custom_layer(t, noop_layer_creator);
        if (ret != 0)
        {
            LOGE("register_custom_layer(%s) failed: %d", t, ret);
        }
    }
}

int PicoDet::load(const char* parampath, const char* modelpath,
                  int num_class_hint, bool use_fp16, bool use_gpu)
{
    picodet_net.clear();
    num_class = std::max(1, num_class_hint);

    picodet_net.opt.use_fp16_packed = use_fp16;
    picodet_net.opt.use_fp16_storage = use_fp16;
    picodet_net.opt.use_fp16_arithmetic = use_fp16;
    picodet_net.opt.num_threads = ncnn::get_big_cpu_count();
#if NCNN_VULKAN
    picodet_net.opt.use_vulkan_compute = use_gpu;
#endif

    register_stubs(picodet_net);

    int ret = picodet_net.load_param(parampath);
    if (ret != 0)
    {
        LOGE("load_param failed: %s, ret=%d", parampath, ret);
        return -1;
    }
    ret = picodet_net.load_model(modelpath);
    if (ret != 0)
    {
        LOGE("load_model failed: %s, ret=%d", modelpath, ret);
        return -2;
    }

    LOGD("Model loaded (file): num_class_hint=%d, target_size=%d", num_class, target_size);
    return 0;
}

int PicoDet::load(AAssetManager* mgr, const char* parampath, const char* modelpath,
                  int num_class_hint, bool use_fp16, bool use_gpu)
{
    picodet_net.clear();
    num_class = std::max(1, num_class_hint);

    picodet_net.opt.use_fp16_packed = use_fp16;
    picodet_net.opt.use_fp16_storage = use_fp16;
    picodet_net.opt.use_fp16_arithmetic = use_fp16;
    picodet_net.opt.num_threads = ncnn::get_big_cpu_count();
#if NCNN_VULKAN
    picodet_net.opt.use_vulkan_compute = use_gpu;
#endif

    register_stubs(picodet_net);

    int ret = picodet_net.load_param(mgr, parampath);
    if (ret != 0)
    {
        LOGE("load_param (asset) failed: %s, ret=%d", parampath, ret);
        return -1;
    }
    ret = picodet_net.load_model(mgr, modelpath);
    if (ret != 0)
    {
        LOGE("load_model (asset) failed: %s, ret=%d", modelpath, ret);
        return -2;
    }

    LOGD("Model loaded (asset): num_class_hint=%d, target_size=%d", num_class, target_size);
    return 0;
}

void PicoDet::set_target_size(int /*_target_size*/)
{
    // The shipped model bakes 320x320 anchors. Forcing 320 keeps decoding correct.
    target_size = 320;
}

void PicoDet::set_prob_threshold(float t) { prob_threshold = t; }
void PicoDet::set_nms_threshold(float t)  { nms_threshold = t; }

int PicoDet::detect(const cv::Mat& rgb, std::vector<DetObject>& objects)
{
    objects.clear();

    // Anti-spoof: if enabled, check if image is from screen first
    if (anti_spoof_enabled)
    {
        if (is_from_screen(rgb))
        {
            LOGD("detect: anti-spoof triggered — image appears to be from screen, skipping detection");
            return 1; // return 1 = spoof detected, no objects
        }
    }

    const int img_w = rgb.cols;
    const int img_h = rgb.rows;

    // ----- letterbox to 320x320 (top-left aligned, pad bottom/right) -----
    int w = img_w;
    int h = img_h;
    float scale;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(img_h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(img_w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    const int wpad = target_size - w;
    const int hpad = target_size - h;
    ncnn::Mat in_pad;
    if (wpad > 0 || hpad > 0)
    {
        ncnn::copy_make_border(in, in_pad,
                               0, hpad, 0, wpad,
                               ncnn::BORDER_CONSTANT, 0.f);
    }
    else
    {
        in_pad = in;
    }

    // PaddleDetection PicoDet standard normalization (RGB order)
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {0.017125f, 0.017507f, 0.017429f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    // ----- in0 carries [h, w, scale_h, scale_w] for the (now-skipped) tail.
    //        We still feed it because the early graph references in0 via Slice. -----
    ncnn::Mat in0(4);
    {
        float* p = (float*)in0.data;
        p[0] = (float)target_size;
        p[1] = (float)target_size;
        p[2] = scale;
        p[3] = scale;
    }

    // ----- inference: extract intermediate blobs only -----
    ncnn::Extractor ex = picodet_net.create_extractor();
    ex.input("in0", in0);
    ex.input("in1", in_pad);

    ncnn::Mat scores; // expected: [w=num_anchors, h=num_class]
    ncnn::Mat boxes;  // expected: [w=4, h=num_anchors]
    int r1 = ex.extract("317", scores);
    int r2 = ex.extract("339", boxes);
    if (r1 != 0 || r2 != 0)
    {
        LOGE("extract failed: scores ret=%d boxes ret=%d", r1, r2);
        return -1;
    }

    LOGD("scores dims: w=%d h=%d c=%d", scores.w, scores.h, scores.c);
    LOGD("boxes  dims: w=%d h=%d c=%d", boxes.w, boxes.h, boxes.c);

    // Resolve layout. Param shows scores=[w=num_anchors, h=num_class]
    // and boxes=[w=4, h=num_anchors]; we assert and read accordingly.
    int num_anchors = 0;
    int nc = 0;
    if (scores.h > 0 && scores.w > 0)
    {
        nc = scores.h;
        num_anchors = scores.w;
    }
    else
    {
        LOGE("unexpected scores shape");
        return -2;
    }
    if (boxes.w != 4 || boxes.h != num_anchors)
    {
        LOGE("unexpected boxes shape: w=%d h=%d (expected w=4 h=%d)",
             boxes.w, boxes.h, num_anchors);
        return -3;
    }

    if (num_class < nc)
    {
        // The user's hint understated the actual class count; trust the model.
        LOGD("num_class hint=%d but model has %d, using %d", num_class, nc, nc);
    }
    // Always use the actual class count from the blob shape
    num_class = nc;
    const int class_count = nc;

    // ----- collect proposals above prob_threshold -----
    std::vector<DetObject> proposals;
    proposals.reserve(256);

    for (int a = 0; a < num_anchors; a++)
    {
        // best class score for this anchor
        int best_label = 0;
        float best_score = -FLT_MAX;
        for (int k = 0; k < class_count; k++)
        {
            // scores layout: row k is class k, length num_anchors
            float s = at2d(scores, k, a);
            if (s > best_score)
            {
                best_score = s;
                best_label = k;
            }
        }
        if (best_score < prob_threshold) continue;

        // boxes layout: row a, 4 cols xyxy
        const float* row = boxes.row(a);
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];

        if (x2 <= x1 || y2 <= y1) continue;

        DetObject obj;
        obj.label = best_label;
        obj.prob = best_score;
        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;
        proposals.push_back(obj);
    }

    // ----- sort + class-aware NMS -----
    qsort_descent(proposals);
    std::vector<int> picked;
    nms_class_aware(proposals, picked, nms_threshold);

    // ----- inverse letterbox back to original image -----
    objects.reserve(picked.size());
    for (size_t i = 0; i < picked.size(); i++)
    {
        DetObject obj = proposals[picked[i]];

        float x1 = obj.rect.x / scale;
        float y1 = obj.rect.y / scale;
        float x2 = (obj.rect.x + obj.rect.width) / scale;
        float y2 = (obj.rect.y + obj.rect.height) / scale;

        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(img_w - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(img_h - 1)), 0.f);

        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;
        if (obj.rect.width > 0 && obj.rect.height > 0)
            objects.push_back(obj);
    }

    LOGD("detect: anchors=%d proposals=%d kept=%d (prob>=%.2f, nms=%.2f)",
         num_anchors, (int)proposals.size(), (int)objects.size(),
         prob_threshold, nms_threshold);

    return 0;
}


// ============================================================================
// Anti-spoof: Detect if image is from a screen/monitor
// Uses moiré pattern detection via FFT frequency analysis
// ============================================================================

bool PicoDet::is_from_screen(const cv::Mat& rgb)
{
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);

    // Resize to fixed size for consistent analysis
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(256, 256));

    // Convert to float for DFT
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F);

    // Apply DFT
    cv::Mat dft_result;
    cv::dft(floatImg, dft_result, cv::DFT_COMPLEX_OUTPUT);

    // Shift zero-frequency to center
    int cx = dft_result.cols / 2;
    int cy = dft_result.rows / 2;

    // Split into real and imaginary
    std::vector<cv::Mat> planes;
    cv::split(dft_result, planes);

    // Compute magnitude
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    // Log scale
    magnitude += cv::Scalar::all(1);
    cv::log(magnitude, magnitude);

    // Normalize
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

    // Analyze high-frequency content
    // Screen moiré creates peaks in mid-to-high frequency bands
    // Measure energy in high-frequency ring (exclude DC center and very high noise)
    float high_freq_energy = 0;
    float mid_freq_energy = 0;
    float total_energy = 0;
    int high_count = 0;
    int mid_count = 0;
    int total_count = 0;

    for (int y = 0; y < magnitude.rows; y++)
    {
        for (int x = 0; x < magnitude.cols; x++)
        {
            float dist = std::sqrt((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
            float val = magnitude.at<float>(y, x);
            total_energy += val;
            total_count++;

            // Mid frequency band (20-60% of max radius)
            if (dist > cx * 0.2f && dist < cx * 0.6f)
            {
                mid_freq_energy += val;
                mid_count++;
            }
            // High frequency band (60-90% of max radius)
            if (dist > cx * 0.6f && dist < cx * 0.9f)
            {
                high_freq_energy += val;
                high_count++;
            }
        }
    }

    // Compute ratios
    float avg_mid = mid_count > 0 ? mid_freq_energy / mid_count : 0;
    float avg_high = high_count > 0 ? high_freq_energy / high_count : 0;
    float avg_total = total_count > 0 ? total_energy / total_count : 0;

    // Moiré pattern indicator: high ratio of mid+high frequency to total
    // Real-world images have most energy in low frequencies
    // Screen images have elevated mid/high frequency due to pixel grid
    float screen_score = (avg_mid + avg_high) / (avg_total + 1e-6f);

    // Also check for periodic peaks (moiré is periodic)
    // Count pixels above threshold in high-freq band
    int peak_count = 0;
    float peak_threshold = avg_high * 2.0f;
    for (int y = 0; y < magnitude.rows; y++)
    {
        for (int x = 0; x < magnitude.cols; x++)
        {
            float dist = std::sqrt((float)((x - cx) * (x - cx) + (y - cy) * (y - cy)));
            if (dist > cx * 0.4f && dist < cx * 0.85f)
            {
                if (magnitude.at<float>(y, x) > peak_threshold)
                    peak_count++;
            }
        }
    }

    float peak_ratio = (float)peak_count / (float)(magnitude.rows * magnitude.cols);

    LOGD("anti-spoof: screen_score=%.4f peak_ratio=%.4f", screen_score, peak_ratio);

    // Thresholds (tuned empirically)
    // screen_score > 1.5 OR peak_ratio > 0.05 → likely from screen
    bool is_screen = (screen_score > 1.5f) || (peak_ratio > 0.05f);

    return is_screen;
}

// ============================================================================
// Draw detections onto RGB image
// ============================================================================

void PicoDet::draw_detections(cv::Mat& rgb, const std::vector<DetObject>& objects)
{
    // Class names and colors (BGR for OpenCV)
    static const char* class_names[] = {"LCK", "SCR"};
    static const cv::Scalar class_colors[] = {
        cv::Scalar(0, 255, 0),   // LCK = green (RGB)
        cv::Scalar(128, 0, 128), // SCR = purple (RGB)
    };
    static const int num_known = 2;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const DetObject& obj = objects[i];

        cv::Scalar color;
        const char* name;
        if (obj.label >= 0 && obj.label < num_known)
        {
            color = class_colors[obj.label];
            name = class_names[obj.label];
        }
        else
        {
            color = cv::Scalar(255, 0, 0); // red fallback
            name = "?";
        }

        // Draw rectangle
        int x1 = (int)obj.rect.x;
        int y1 = (int)obj.rect.y;
        int x2 = (int)(obj.rect.x + obj.rect.width);
        int y2 = (int)(obj.rect.y + obj.rect.height);

        cv::rectangle(rgb, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Draw label background + text
        char text[64];
        snprintf(text, sizeof(text), "%s %.0f%%", name, obj.prob * 100.f);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int label_y = std::max(y1 - label_size.height - 4, 0);
        cv::rectangle(rgb,
                      cv::Point(x1, label_y),
                      cv::Point(x1 + label_size.width + 4, label_y + label_size.height + baseLine + 4),
                      color, -1); // filled

        cv::putText(rgb, text,
                    cv::Point(x1 + 2, label_y + label_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}
