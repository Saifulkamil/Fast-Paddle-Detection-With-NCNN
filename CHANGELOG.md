# Changelog

## 0.0.2

### New Features
- 🔀 **Multi-Model Support**: Auto-detects model format at load time
  - Format A: Models with baked-in post-processing (finetune/custom)
  - Format C: Models without post-processing (COCO pretrained, recommended)
- 🧠 **FPN Decode**: Full DFL softmax + expectation decode for no-postprocess models
- 🏷️ **Dynamic Labels**: Auto-selects COCO 80 labels or custom 2-class (LCK/SCR) based on model
- 🎛️ **Runtime Model Loading**: Pick .param and .bin separately from device storage
- ⚡ **GPU Toggle**: Switch CPU/GPU (Vulkan) from settings with auto-detection
- 🛡️ **Anti-Spoof**: FFT moiré pattern detection to reject screen/monitor images
- � **Orientation Aware**: Camera preview rotates with device physical orientation
- � **Capture Guard**: Photo capture only enabled when objects are detected
- ▶️ **Start/Stop Camera**: Manual control over detection pipeline

### Improvements
- Optimized pipeline: reduced from 4 bitmap copies to 1 per frame
- On-demand capture allocation (no per-frame overhead)
- Auto num_class detection from model blob shape
- Separate .param and .bin file picker buttons
- GPU availability check before enabling toggle

### Bug Fixes
- Fixed bbox position mismatch in camera (draw in native C++)
- Fixed threshold not applying to all classes equally
- Fixed capture crash (recycled bitmap race condition)
- Fixed camera black screen (runtime permission)
- Fixed landscape detection (OrientationEventListener + targetRotation)
- Fixed preview flip when device rotated
- Fixed GPU switch not responsive in settings sheet
- Fixed model load failure due to mismatched .bin filename
- Fixed Resize layer not supported (replaced with Interp in .param)
- Fixed num_class showing 1600 instead of 80 (min dimension heuristic)
- Fixed labels showing "person" for custom 2-class model

### Model Support
- PP-PicoDet S/M/L (320×320)
- COCO 80-class pretrained models
- Custom finetune models (2+ classes)
- ONNX → NCNN via PNNX conversion documented

## 0.0.1

- Initial project setup
- Basic NCNN inference pipeline
- JNI bridge + Kotlin plugin
- Dart API + method channel
