// PicoDet NCNN Detection Engine
//
// This wrapper consumes a PaddleDetection PicoDet model exported via PNNX
// that bakes detection post-processing (decode + NMS + TopK) into the graph.
// ncnn cannot run those tail ops directly, so we:
//   1. Register no-op stubs for the unknown layer types so load_param succeeds.
//   2. Extract intermediate blobs that lie BEFORE the unknown layers:
//        - "317" : per-anchor class scores  [w=num_anchors, h=num_class]
//        - "339" : decoded boxes (xyxy) in network input pixel space
//                  [w=4, h=num_anchors]
//   3. Run threshold + class-aware NMS in C++, then inverse-letterbox.
//
// The model bakes anchors at 320x320, so target_size is fixed at 320.

#ifndef PPDET_PICO_H
#define PPDET_PICO_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <net.h>

struct DetObject
{
    cv::Rect_<float> rect; // x, y, width, height (absolute pixels in original image)
    int label;
    float prob;
};

class PicoDet
{
public:
    PicoDet();
    ~PicoDet();

    int load(const char* parampath, const char* modelpath,
             int num_class_hint, bool use_fp16 = true, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* parampath, const char* modelpath,
             int num_class_hint, bool use_fp16 = true, bool use_gpu = false);

    void set_target_size(int target_size);
    void set_prob_threshold(float threshold);
    void set_nms_threshold(float threshold);

    int get_num_class() const { return num_class; }

    int detect(const cv::Mat& rgb, std::vector<DetObject>& objects);

    // Draw bounding boxes directly onto an RGB image (modifies in-place).
    static void draw_detections(cv::Mat& rgb, const std::vector<DetObject>& objects);

protected:
    ncnn::Net picodet_net;
    int target_size;
    int num_class;
    float prob_threshold;
    float nms_threshold;
};

#endif // PPDET_PICO_H
