# Paddle Detection 🎯

[![License](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](LICENSE)

A high-performance Flutter plugin for **offline object detection** on Android using **PP-PicoDet** (PaddleDetection) and **NCNN**.

This plugin supports real-time detection directly from the camera feed, photo capture with bounding box visualization, and on-device inference. It processes everything locally using C++ (NCNN), meaning it is **extremely fast** and requires **zero internet connection**.

## ✨ Key Features

- ⚡ **Real-time Detection**: Detect objects instantly from the live camera preview.
- 📸 **Photo Capture**: Save both clean and annotated (with bbox) images on detection.
- 🔦 **Flash Control**: Toggle camera flash/torch on/off.
- 🔄 **Switch Camera**: Toggle between front and back camera.
- 📱 **Orientation Aware**: Camera preview rotates with device physical orientation (app stays portrait).
- 🧠 **Auto Class Detection**: Number of classes read from model blob shape automatically.
- ⚡ **GPU Acceleration**: Optional Vulkan GPU inference with auto-detection of GPU availability.
- 📴 **100% Offline**: Uses local NCNN models. No API calls or cloud dependencies.
- 🔋 **Optimized Performance**: Minimal bitmap copies per frame, on-demand capture allocation.

---

## 🎥 Demo

*(Replace placeholder links with your uploaded media)*

<table align="center">
  <tr>
    <td align="center"><b>Real-Time Detection</b></td>
    <td align="center"><b>Photo Capture Result</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="#" alt="Realtime Detection" width="250">
    </td>
    <td align="center">
      <img src="#" alt="Photo Capture" width="250">
    </td>
  </tr>
</table>

---

## 📱 Platform Support

| Platform    | Support | Note                                             |
| :---------- | :-----: | :----------------------------------------------- |
| **Android** |   ✅    | Fully supported (Requires Android 7.0 / API 24+) |
| **iOS**     |   ❌    | Not supported                                    |
| **Web**     |   ❌    | Not supported                                    |
| **Desktop** |   ❌    | Not supported                                    |

---

## ⚙️ Android Setup (Required)

### 1. Minimum SDK Version

Ensure your `android/app/build.gradle.kts` has `minSdk` of at least `24`:

```kotlin
android {
    defaultConfig {
        minSdk = 24
    }
}
```

### 2. NDK Version

Set NDK version to match the plugin:

```kotlin
android {
    ndkVersion = "29.0.14206865"
}
```

### 3. Reduce APK Size (Crucial) 🚨

NCNN libraries are large. Filter architectures to only physical ARM devices:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }
}
```

### 4. Camera Permission

Add to your `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-feature android:name="android.hardware.camera" android:required="false"/>
```

Request permission at runtime using `permission_handler` package.

### 5. Model Files

Place your NCNN model files in `android/app/src/main/assets/`:

```
android/app/src/main/assets/
├── model.ncnn.param
└── model.ncnn.bin
```

---

## 🚀 Usage

### 1. Import

```dart
import 'package:paddle_detection/paddle_detection.dart';
```

### 2. Load Model

```dart
final detector = PaddleDetection();

// Load from assets (default CPU)
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
);
print('Classes: ${info.numClass}'); // auto-detected from model

// Or load with GPU
final hasGpu = await detector.hasGpu();
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
  cpuGpu: hasGpu ? 1 : 0, // 0=CPU, 1=GPU(Vulkan), 2=GPU(Turnip)
);

// Or load from file path (user-picked)
final info = await detector.loadModelFromFile(
  paramPath: '/path/to/model.ncnn.param',
  binPath: '/path/to/model.ncnn.bin',
);
```

### 3. Set Threshold

```dart
// Single value controls both prob and NMS threshold
await detector.setThreshold(threshold: 0.8);
```

### 4. Real-time Camera Detection

```dart
// Start camera (returns texture info for display)
final cam = await detector.startCamera();

// Display in Flutter
Texture(textureId: cam.textureId)

// Listen to detection stream
detector.detectionStream.listen((event) {
  print('${event.detections.length} objects, ${event.inferenceMs}ms');
  print('Device rotation: ${event.deviceRotation}°');
});

// Toggle flash
await detector.toggleFlash(enable: true);

// Switch front/back camera
await detector.switchCamera();

// Stop camera
await detector.stopCamera();
```

### 5. Capture Photo

Capture saves both a clean image and an annotated image (with bbox drawn):

```dart
final result = await detector.capturePhoto(
  folder: '/storage/emulated/0/DCIM/MyApp',
  prefix: 'detection',
);

print(result.cleanPath);     // /path/detection_1234567890.jpg
print(result.annotatedPath); // /path/detection_1234567890_bbox.jpg
print(result.detections);    // List<DetectionResult>
```

> **Note**: Capture button is only active when objects are detected.

### 6. Detect from Image File (Gallery)

```dart
final results = await detector.detect('/path/to/image.jpg');
for (final det in results) {
  print('${det.label} ${det.probability} ${det.x},${det.y},${det.width},${det.height}');
}
```

### 7. GPU Detection

```dart
final hasGpu = await detector.hasGpu();     // true if Vulkan available
final gpuCount = await detector.getGpuCount(); // number of GPU devices
```

### 8. Cleanup

```dart
await detector.stopCamera();
await detector.dispose();
```

---

## 📝 API Reference

| Method | Description |
| :----- | :---------- |
| `loadModel(paramName, binName, cpuGpu)` | Load model from Android assets |
| `loadModelFromFile(paramPath, binPath, cpuGpu)` | Load model from absolute file paths |
| `setThreshold(threshold)` | Set detection threshold (0.0–1.0) |
| `getNumClass()` | Get number of classes from loaded model |
| `hasGpu()` | Check if Vulkan GPU is available |
| `getGpuCount()` | Get number of GPU devices |
| `detect(imagePath)` | Run detection on image file |
| `detectFromBytes(data, width, height)` | Run detection on raw RGBA bytes |
| `startCamera()` | Start native camera, returns texture info |
| `stopCamera()` | Stop native camera |
| `switchCamera()` | Toggle front/back camera |
| `toggleFlash(enable)` | Toggle camera flash |
| `capturePhoto(folder, prefix)` | Capture clean + annotated images |
| `detectionStream` | Stream of realtime detection events |
| `dispose()` | Release all native resources |

### Data Classes

```dart
class ModelInfo {
  final bool success;
  final int numClass; // auto-detected from model
}

class DetectionResult {
  final int label;
  final double probability;
  final double x, y, width, height; // absolute pixels
}

class CameraInfo {
  final int textureId;
  final int previewWidth, previewHeight;
}

class CameraDetectionEvent {
  final List<DetectionResult> detections;
  final int imageWidth, imageHeight;
  final int inferenceMs;
  final int deviceRotation; // 0, 90, 180, 270
}

class CaptureResult {
  final String cleanPath;     // image without bbox
  final String annotatedPath; // image with bbox
  final List<DetectionResult> detections;
}
```

---

## 🧠 Model Specification (Important!)

This plugin expects a specific NCNN model format. If you want to use your own model or train a custom one, follow this specification.

### Model Architecture

The plugin is designed for **PP-PicoDet** models exported from PaddleDetection via PNNX/ONNX to NCNN format, with **post-processing baked into the graph**.

### Input Specification

The model must have **2 input blobs**:

| Blob Name | Shape | Description |
|-----------|-------|-------------|
| `in0` | `[4]` | Metadata: `[target_height, target_width, scale_factor_h, scale_factor_w]` |
| `in1` | `[3, 320, 320]` | Normalized RGB image tensor (after letterbox + normalize) |

**Preprocessing (handled by plugin):**
1. Resize image keeping aspect ratio to fit 320×320
2. Pad (letterbox) to exactly 320×320 with zeros
3. Normalize: `mean = [123.675, 116.28, 103.53]`, `norm = [0.017125, 0.017507, 0.017429]`
4. `in0` is filled with `[320, 320, scale_h, scale_w]` where scale = 320/max(img_w, img_h)

### Output Specification (Intermediate Blobs)

The plugin extracts **intermediate blobs** (not final output), because the model's built-in post-processing uses ops that ncnn doesn't support natively:

| Blob Name | Shape | Description |
|-----------|-------|-------------|
| `"317"` | `[w=num_anchors, h=num_class]` | Per-anchor class scores (already sigmoid-activated) |
| `"339"` | `[w=4, h=num_anchors]` | Decoded boxes in xyxy format (in 320×320 input pixel space, multiplied by stride factor) |

**For the default PicoDet model:**
- `num_anchors = 2125` (sum of all FPN levels: 40×40 + 20×20 + 10×10 + 5×5 = 1600+400+100+25)
- `num_class = 2` (detected from blob shape automatically)

### Post-processing (handled by plugin in C++)

1. For each anchor: find best class score → filter by `prob_threshold`
2. Read box coordinates from blob 339: `[x1, y1, x2, y2]`
3. Class-aware NMS with `nms_threshold`
4. Inverse letterbox: divide coordinates by scale factor to get original image pixels

### Unsupported Layers (Handled by No-op Stubs)

The model graph contains these layers that ncnn doesn't support. The plugin registers no-op stubs for them so `load_param` succeeds:

- `pnnx.Expression`
- `NonMaxSuppression`
- `TopK`
- `Gather`
- `F.embedding`
- `Tensor.to`

These are all in the post-processing tail and are never executed (ncnn evaluates lazily).
 

### Model Compatibility Checklist

| Requirement | ✓ |
|-------------|---|
| PicoDet architecture (PP-PicoDet-S/M/L) | Required |
| Input size 320×320 | Required (hardcoded in plugin) |
| Post-processing baked in (NMS, TopK, etc.) | Required |
| Exported via PNNX from ONNX | Required |
| 2 input blobs (`in0` metadata, `in1` pixels) | Required |
| Intermediate score blob accessible | Required |
| Intermediate box blob accessible | Required |

### Changing Blob Names

If your model has different blob numbers for scores/boxes, update in `ppdet_pico.cpp`:

```cpp
// In detect() function:
int r1 = ex.extract("317", scores);  // ← change "317" to your score blob
int r2 = ex.extract("339", boxes);   // ← change "339" to your box blob
```

To find the correct blob numbers, open your `.param` file and look for:
- **Score blob**: the `Concat` that merges all FPN-level class predictions (shape `[num_anchors, num_class]`)
- **Box blob**: the `BinaryOp` (multiply) that produces decoded xyxy boxes (shape `[4, num_anchors]`)

---

## 💡 Tips & Best Practices

1. **Memory Management**: Always call `stopCamera()` and `dispose()` in your widget's `dispose()` method.
2. **Threshold Tuning**: Start with `0.5` and increase to reduce false positives. For production, `0.7–0.9` is typical.
3. **Capture Guard**: The plugin only allows capture when objects are detected — no empty captures.
4. **Flash**: Only available on back camera. Automatically disabled when switching to front.
5. **Model Size**: Use lightweight PicoDet models (PP-PicoDet-S) for mobile. Larger models will be slower.
6. **GPU Mode**: Set `cpuGpu: 1` for Vulkan GPU acceleration on supported devices. CPU (`cpuGpu: 0`) is more stable across all devices. Not all devices are faster with GPU — test both.
7. **Debug FPS**: In debug builds, FPS and inference time are shown as an overlay badge.
8. **Orientation**: Camera preview automatically rotates with device physical orientation while app stays portrait.
9. **Num Classes**: The plugin auto-detects the number of classes from the model's blob shape — no need to specify manually.

---

## 🏗️ Architecture

```
Flutter (Dart API)
    │
    ├── MethodChannel "paddle_detection"
    └── EventChannel "paddle_detection/detections"
          │
          ▼
Kotlin (CameraX + Plugin)
    │
    ├── CameraX ImageAnalysis → frame capture
    ├── OrientationEventListener → dynamic rotation
    ├── Rotate bitmap → JNI call
    ├── Render annotated frame → Flutter Texture
    └── Stream detections + inferenceMs + deviceRotation → Dart
          │
          ▼
C++ / JNI
    │
    ├── NCNN inference (ncnn-20260113-android-vulkan)
    │     ├── No-op stub layers for unsupported ops
    │     └── Extract intermediate blobs (lazy evaluation)
    ├── Post-processing (threshold + NMS + inverse letterbox)
    ├── OpenCV bbox drawing (rectangle + putText)
    └── OpenCV Mobile 4.13.0 (core + imgproc + highgui)
```

---

## 📄 License

This project is licensed under [CC0 1.0 Universal](LICENSE) — public domain dedication.
