# Paddle Detection 🎯

[![License](https://img.shields.io/badge/License-CC0_1.0-blue.svg)](LICENSE)

A high-performance Flutter plugin for **offline object detection** on Android using **PP-PicoDet** (PaddleDetection) and **NCNN**.

This plugin supports real-time detection directly from the camera feed, photo capture with bounding box visualization, and on-device inference. It processes everything locally using C++ (NCNN), meaning it is **extremely fast** and requires **zero internet connection**.

## ✨ Key Features

- ⚡ **Real-time Detection**: Detect objects instantly from the live camera preview.
- 🎯 **Native C++ Bbox Drawing**: Bounding boxes drawn directly on the frame in C++ — zero coordinate mapping issues.
- 📸 **Photo Capture**: Save both clean and annotated (with bbox) images on detection.
- 🔦 **Flash Control**: Toggle camera flash/torch on/off.
- 🔄 **Switch Camera**: Toggle between front and back camera.
- 🎛️ **Single Threshold Control**: One knob controls both prob and NMS threshold.
- 🧠 **Auto Class Detection**: Number of classes read from model blob shape automatically.
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

// Load from assets (default)
final info = await detector.loadModel(
  paramName: 'model.ncnn.param',
  binName: 'model.ncnn.bin',
);
print('Classes: ${info.numClass}'); // auto-detected from model

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

### 7. Cleanup

```dart
await detector.stopCamera();
await detector.dispose();
```

---

## 📝 API Reference

| Method | Description |
| :----- | :---------- |
| `loadModel(paramName, binName)` | Load model from Android assets |
| `loadModelFromFile(paramPath, binPath)` | Load model from absolute file paths |
| `setThreshold(threshold)` | Set detection threshold (0.0–1.0) |
| `getNumClass()` | Get number of classes from loaded model |
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
// Model info after loading
class ModelInfo {
  final bool success;
  final int numClass;
}

// Single detection result
class DetectionResult {
  final int label;
  final double probability;
  final double x, y, width, height; // absolute pixels
}

// Camera info
class CameraInfo {
  final int textureId;
  final int previewWidth, previewHeight;
}

// Realtime stream event
class CameraDetectionEvent {
  final List<DetectionResult> detections;
  final int imageWidth, imageHeight;
  final int inferenceMs;
}

// Capture result
class CaptureResult {
  final String cleanPath;
  final String annotatedPath;
  final List<DetectionResult> detections;
}
```

---

## 💡 Tips & Best Practices

1. **Memory Management**: Always call `stopCamera()` and `dispose()` in your widget's `dispose()` method.
2. **Threshold Tuning**: Start with `0.5` and increase to reduce false positives. For production, `0.7–0.9` is typical.
3. **Capture Guard**: The plugin only allows capture when objects are detected — no empty captures.
4. **Flash**: Only available on back camera. Automatically disabled when switching to front.
5. **Model Size**: Use lightweight PicoDet models (PP-PicoDet-S) for mobile. Larger models will be slower.
6. **GPU Mode**: Set `cpuGpu: 1` for Vulkan GPU acceleration on supported devices. CPU (`cpuGpu: 0`) is more stable across all devices.
7. **Debug FPS**: In debug builds, FPS and inference time are shown as an overlay badge.

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
    ├── Rotate bitmap → JNI call
    ├── Render annotated frame → Flutter Texture
    └── Stream detections → Dart
          │
          ▼
C++ / JNI
    │
    ├── NCNN inference (ncnn-20260113-android-vulkan)
    ├── Post-processing (threshold + NMS)
    ├── OpenCV bbox drawing (rectangle + putText)
    └── OpenCV Mobile 4.13.0 (core + imgproc + highgui)
```

---

## 📄 License

This project is licensed under [CC0 1.0 Universal](LICENSE) — public domain dedication.
