# Changelog

## 1.0.0

### Features
- ⚡ Real-time object detection using PP-PicoDet + NCNN
- 📸 Native camera (CameraX) with Flutter Texture rendering
- 🔦 Flash on/off control
- 🔄 Switch front/back camera
- 📱 Orientation-aware preview (rotates with device, app stays portrait)
- 📷 Photo capture (clean + annotated with bbox)
- 🖼️ Gallery image detection (pick from gallery)
- ⚡ GPU acceleration (Vulkan) with auto-detection
- 🛡️ Anti-spoof detection (moiré pattern FFT analysis)
- 🧠 Auto class count detection from model blob shape
- 🔧 Custom model loading from file path
- 📊 Inference time + FPS display (debug mode)

### Architecture
- Native camera via CameraX ImageAnalysis (no Flutter camera plugin)
- NCNN inference with no-op stub layers for unsupported ops
- Intermediate blob extraction (lazy evaluation)
- C++ post-processing: threshold + class-aware NMS + inverse letterbox
- Optimized pipeline: 1 bitmap copy per frame (rotate only)
- On-demand capture allocation (no per-frame overhead)
- OrientationEventListener for dynamic rotation handling
- EventChannel streaming (detections + inferenceMs + deviceRotation)

### Model Support
- PP-PicoDet (S/M/L) exported via PNNX with baked-in post-processing
- Input: 320×320 fixed (letterbox + normalize)
- 2 input blobs: `in0` (metadata), `in1` (pixel tensor)
- Intermediate blobs: `317` (scores), `339` (boxes)
- Stub layers: pnnx.Expression, NonMaxSuppression, TopK, Gather, F.embedding, Tensor.to

### Platform
- Android only (API 24+)
- arm64-v8a + armeabi-v7a
- NDK 29, NCNN 20260113, OpenCV Mobile 4.13.0

## 0.0.1

- Initial project setup
