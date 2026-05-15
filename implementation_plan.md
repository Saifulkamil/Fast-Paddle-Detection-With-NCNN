# 📋 Implementation Plan — Plugin `paddle_detection` (PP-PicoDet + NCNN)

> Dokumen ini mencatat status implementasi plugin Flutter `paddle_detection` yang menggunakan PP-PicoDet dengan runtime NCNN untuk object detection di Android.

---

## Arsitektur Final

```
Flutter (Dart)
  │
  ├── PaddleDetection class (paddle_detection.dart)
  │     ├── loadModel / loadModelFromFile
  │     ├── detect / detectFromBytes
  │     ├── setThreshold
  │     ├── startCamera / stopCamera / switchCamera
  │     ├── toggleFlash
  │     ├── capturePhoto
  │     └── detectionStream (EventChannel)
  │
  ├── MethodChannel "paddle_detection"
  └── EventChannel "paddle_detection/detections"
        │
        ▼
Kotlin (PaddleDetectionPlugin.kt)
  │
  ├── CameraX (ImageAnalysis + lifecycle)
  │     ├── Frame capture → rotate → nativeDetectAndDraw
  │     ├── Render annotated bitmap ke Flutter Texture
  │     └── Stream detections + inferenceMs ke Dart
  │
  ├── Flash control (CameraControl.enableTorch)
  ├── Switch camera (front/back)
  ├── Capture photo (clean + annotated JPEG)
  │
  └── JNI calls
        │
        ▼
C++ (paddle_detection_jni.cpp + ppdet_pico.cpp)
  │
  ├── NCNN runtime (ncnn-20260113-android-vulkan)
  │     ├── No-op stub layers (pnnx.Expression, NonMaxSuppression, TopK, Gather, F.embedding, Tensor.to)
  │     ├── Extract blob "317" (class scores) + blob "339" (decoded boxes)
  │     └── Lazy evaluation — stub layers never executed
  │
  ├── Post-processing di C++
  │     ├── Threshold filtering (prob_threshold)
  │     ├── Class-aware NMS (nms_threshold)
  │     └── Inverse letterbox (map ke original image coords)
  │
  ├── draw_detections() — OpenCV rectangle + putText langsung di bitmap
  │
  └── OpenCV Mobile 4.13.0 (core + imgproc + highgui)
```

---

## Model Strategy

Model `model.ncnn.param` / `model.ncnn.bin` di-export dari PaddleDetection dengan **post-processing baked-in** (NMS, TopK, Gather, dll). ncnn tidak punya layer-layer tersebut secara native.

**Solusi yang diterapkan:**
1. Register no-op stub layer untuk 6 jenis layer asing
2. Extract intermediate blobs **sebelum** layer asing:
   - Blob `"317"` → class scores `[w=num_anchors, h=num_class]`
   - Blob `"339"` → decoded boxes xyxy `[w=4, h=num_anchors]`
3. C++ melakukan threshold + NMS + inverse letterbox sendiri
4. `num_class` dibaca dari blob shape (bukan hardcode)

**Input model:**
- `in0` = `[target_size, target_size, scale_h, scale_w]` (metadata)
- `in1` = normalized pixel tensor 320×320

---

## Status Implementasi

### ✅ Selesai

| # | Fitur | Status |
|---|-------|--------|
| 1 | Build system (NDK 29, CMake, CameraX deps) | ✅ |
| 2 | NCNN + OpenCV Mobile linked | ✅ |
| 3 | No-op stub layers untuk model asing | ✅ |
| 4 | Inference dari blob menengah (317, 339) | ✅ |
| 5 | Threshold + NMS di C++ | ✅ |
| 6 | Draw bbox di C++ (OpenCV) | ✅ |
| 7 | JNI bridge (detect, detectAndDraw, detectFromPath, detectFromBytes) | ✅ |
| 8 | Kotlin plugin (MethodChannel + EventChannel) | ✅ |
| 9 | Native camera (CameraX ImageAnalysis → Texture) | ✅ |
| 10 | Flash on/off | ✅ |
| 11 | Switch camera (front/back) | ✅ |
| 12 | Capture photo (clean + annotated, on-demand copy) | ✅ |
| 13 | Dart API lengkap (loadModel, detect, camera, capture, stream) | ✅ |
| 14 | Example app (camera tab + gallery tab + settings) | ✅ |
| 15 | Single threshold (prob = nms = value) | ✅ |
| 16 | Auto-load model saat app buka | ✅ |
| 17 | Capture hanya aktif saat ada deteksi | ✅ |
| 18 | Start/stop camera manual | ✅ |
| 19 | FPS + inference ms (debug mode only) | ✅ |
| 20 | Performa optimasi (reduce bitmap copy per frame) | ✅ |
| 21 | num_class dari blob shape model | ✅ |
| 22 | Class labels (LCK=green, SCR=purple) di C++ | ✅ |

---

## Konfigurasi

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `target_size` | 320 | Fixed (model bakes 320×320 anchors) |
| `cpuGpu` | 0 (CPU) | 0=CPU, 1=GPU Vulkan, 2=Turnip |
| `prob_threshold` | 0.8 | Dari Dart `setThreshold(threshold: 0.8)` |
| `nms_threshold` | 0.8 | Sama dengan prob (single knob) |
| `num_threads` | `ncnn::get_big_cpu_count()` | Otomatis dari ncnn |
| Camera resolution | 640×480 (landscape sensor) | Rotate ke 480×640 portrait |
| Camera default | Back | Bisa switch via `switchCamera()` |

---

## File Structure

```
paddle_detection/
├── android/
│   ├── build.gradle                          (CameraX deps, NDK 29, CMake)
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── jni/
│       │   ├── CMakeLists.txt                (ncnn + opencv + jnigraphics)
│       │   ├── ppdet_pico.h                  (PicoDet class + DetObject)
│       │   ├── ppdet_pico.cpp                (stub layers + inference + draw)
│       │   ├── paddle_detection_jni.cpp      (JNI bridge)
│       │   ├── ncnn-20260113-android-vulkan/ (prebuilt ncnn)
│       │   └── opencv-mobile-4.13.0-android/ (prebuilt opencv)
│       └── kotlin/.../PaddleDetectionPlugin.kt (CameraX + MethodChannel)
├── lib/
│   ├── paddle_detection.dart                 (Public API + CameraInfo + CaptureResult)
│   ├── paddle_detection_platform_interface.dart
│   ├── paddle_detection_method_channel.dart
│   └── src/detection_result.dart
└── example/
    ├── android/app/src/main/
    │   ├── AndroidManifest.xml               (CAMERA permission)
    │   └── assets/
    │       ├── model.ncnn.param
    │       └── model.ncnn.bin
    └── lib/main.dart                         (Camera + Gallery + Settings)
```

---

## Dart API Reference

```dart
final detector = PaddleDetection();

// Load model (returns numClass from model)
final info = await detector.loadModel(paramName: 'model.ncnn.param', binName: 'model.ncnn.bin');
print(info.numClass); // 2

// Set threshold
await detector.setThreshold(threshold: 0.8);

// Single image detection
final results = await detector.detect('/path/to/image.jpg');

// Camera realtime
final cam = await detector.startCamera();
// cam.textureId → use with Texture widget
// cam.previewWidth, cam.previewHeight

detector.detectionStream.listen((event) {
  // event.detections, event.inferenceMs
});

await detector.toggleFlash(enable: true);
await detector.switchCamera(); // toggle front/back

// Capture (only when detections present)
final capture = await detector.capturePhoto(folder: '/path/to/folder', prefix: 'det');
// capture.cleanPath — image tanpa bbox
// capture.annotatedPath — image dengan bbox
// capture.detections — list DetectionResult

await detector.stopCamera();
await detector.dispose();
```

---

## Performa

| Metric | Nilai (Infinix X6853, Helio G88) |
|--------|----------------------------------|
| Inference | ~280ms per frame |
| FPS | ~3-4 FPS |
| Bitmap copies per frame | 1 (rotate only, optimized) |
| Camera resolution | 640×480 → 480×640 portrait |
| Model input | 320×320 |

### Optimasi yang sudah diterapkan:
- Bitmap copy dikurangi dari 4 per frame → 1 (rotate saja)
- Clean copy hanya dibuat saat capture (bukan setiap frame)
- `nativeDetectAndDraw` in-place (tidak buat bitmap baru)
- `STRATEGY_KEEP_ONLY_LATEST` — frame di-drop jika inference belum selesai

### Optimasi potensial (belum diterapkan):
- Turunkan camera resolution ke 320×240 (kurangi pre-processing ~4x)
- GPU inference (Vulkan) — bisa cut inference dari ~280ms → ~80ms
- Pisahkan Preview + ImageAnalysis (smooth 30fps preview + async inference)

---

## Masalah yang Sudah Diselesaikan

| # | Masalah | Solusi |
|---|---------|--------|
| 1 | Gradle cache corrupt | Hapus `~/.gradle/caches/8.12/transforms` |
| 2 | NDK version mismatch | Set `ndkVersion = "29.0.14206865"` di app |
| 3 | minSdk mismatch | Set `minSdk = 24` |
| 4 | `cv::imread` undefined | Tambah `highgui` ke CMake find_package |
| 5 | `pnnx.Expression` not registered | Register no-op stub layers |
| 6 | Model load gagal (unknown layers) | 6 stub layers untuk semua unknown ops |
| 7 | Bbox geser di camera | Draw bbox di native C++ (bukan Flutter overlay) |
| 8 | SCR muncul di bawah threshold | Fix: prob = threshold langsung (bukan -0.10) |
| 9 | numClass 80 (salah) | Baca dari blob shape, bukan hint |
| 10 | Capture crash (recycled bitmap) | Copy bitmap on-demand saat capture |
| 11 | Camera hitam (no permission) | Tambah runtime permission request |
| 12 | Lag (3-4 bitmap copy per frame) | Reduce ke 1 copy (rotate only) |

---

## Catatan Penting

1. **Model ini UNIK** — post-processing baked-in, ncnn tidak bisa run langsung. Solusi: stub layers + extract blob menengah.
2. **Blob "317" dan "339"** adalah satu-satunya blob yang di-extract. Semua blob setelahnya (340+) bergantung pada unknown layers.
3. **`in0` bukan pixel data** — ini metadata `[h, w, scale_h, scale_w]` yang dipakai oleh graph internal model.
4. **opencv-mobile** hanya punya `core`, `imgproc`, `highgui` (versi minimal). Tidak ada `imgcodecs` terpisah — `imread/imwrite` ada di `highgui`.
5. **Flash hanya tersedia di kamera belakang** — otomatis off saat switch ke depan.
6. **Capture on-demand** — clean copy dibuat hanya saat user tap capture, bukan setiap frame. Ini menjaga performa.
