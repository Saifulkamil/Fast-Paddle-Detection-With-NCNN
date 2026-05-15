import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'paddle_detection_platform_interface.dart';
import 'src/detection_result.dart';

export 'src/detection_result.dart';

/// Camera info returned by [startCamera].
class CameraInfo {
  final int textureId;
  final int previewWidth;
  final int previewHeight;
  const CameraInfo({required this.textureId, required this.previewWidth, required this.previewHeight});
}

/// Detection event from native camera stream.
class CameraDetectionEvent {
  final List<DetectionResult> detections;
  final int imageWidth;
  final int imageHeight;
  final int inferenceMs;
  const CameraDetectionEvent({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.inferenceMs,
  });
}

/// Result of a photo capture.
class CaptureResult {
  /// Path to the clean image (no bbox).
  final String cleanPath;

  /// Path to the annotated image (with bbox drawn).
  final String annotatedPath;

  /// Detections at the moment of capture.
  final List<DetectionResult> detections;
  const CaptureResult({required this.cleanPath, required this.annotatedPath, required this.detections});
}

/// Model info returned after loading.
class ModelInfo {
  final bool success;

  /// Number of classes detected from the model.
  final int numClass;
  const ModelInfo({required this.success, required this.numClass});
}

/// Flutter plugin for PaddlePaddle PP-PicoDet object detection using NCNN.
class PaddleDetection {
  static const _methodChannel = MethodChannel('paddle_detection');
  static const _eventChannel = EventChannel('paddle_detection/detections');

  /// Load model from assets. Returns [ModelInfo] with actual class count from model.
  Future<ModelInfo> loadModel({
    required String paramName,
    required String binName,
    int numClass = 2,
    int sizeId = 0,
    int cpuGpu = 0,
  }) async {
    final result = await _methodChannel.invokeMapMethod<String, dynamic>('loadModel', {
      'paramName': paramName,
      'binName': binName,
      'numClass': numClass,
      'sizeId': sizeId,
      'cpuGpu': cpuGpu,
    });
    return ModelInfo(success: result?['success'] == true, numClass: (result?['numClass'] as int?) ?? 0);
  }

  /// Load model from file paths. Returns [ModelInfo] with actual class count.
  Future<ModelInfo> loadModelFromFile({
    required String paramPath,
    required String binPath,
    int numClass = 2,
    int sizeId = 0,
    int cpuGpu = 0,
  }) async {
    final result = await _methodChannel.invokeMapMethod<String, dynamic>('loadModelFromFile', {
      'paramPath': paramPath,
      'binPath': binPath,
      'numClass': numClass,
      'sizeId': sizeId,
      'cpuGpu': cpuGpu,
    });
    return ModelInfo(success: result?['success'] == true, numClass: (result?['numClass'] as int?) ?? 0);
  }

  /// Get number of classes from loaded model.
  Future<int> getNumClass() async {
    final result = await _methodChannel.invokeMethod<int>('getNumClass');
    return result ?? 0;
  }

  /// Detect objects in an image file.
  Future<List<DetectionResult>> detect(String imagePath) {
    return PaddleDetectionPlatform.instance.detect(imagePath: imagePath);
  }

  /// Detect objects from raw RGBA pixel bytes.
  Future<List<DetectionResult>> detectFromBytes(Uint8List data, int width, int height) {
    return PaddleDetectionPlatform.instance.detectFromBytes(data: data, width: width, height: height);
  }

  /// Set detection threshold. Both prob and NMS use the same value.
  Future<void> setThreshold({required double threshold}) {
    final t = threshold.clamp(0.0, 1.0);
    return PaddleDetectionPlatform.instance.setThreshold(probThreshold: t, nmsThreshold: t);
  }

  /// Start native camera. Returns texture info.
  Future<CameraInfo> startCamera() async {
    final result = await _methodChannel.invokeMapMethod<String, dynamic>('startCamera');
    return CameraInfo(
      textureId: result!['textureId'] as int,
      previewWidth: result['previewWidth'] as int,
      previewHeight: result['previewHeight'] as int,
    );
  }

  /// Stop native camera.
  Future<void> stopCamera() async {
    await _methodChannel.invokeMethod('stopCamera');
  }

  /// Toggle flash. Returns new flash state.
  Future<bool> toggleFlash({bool? enable}) async {
    final result = await _methodChannel.invokeMethod<bool>('toggleFlash', {'enable': enable});
    return result ?? false;
  }

  /// Switch between front and back camera. Returns true if now using front camera.
  Future<bool> switchCamera() async {
    final result = await _methodChannel.invokeMethod<bool>('switchCamera');
    return result ?? false;
  }

  /// Capture current frame. Saves clean + annotated images to [folder].
  /// Returns paths and detections.
  Future<CaptureResult> capturePhoto({required String folder, String prefix = 'capture'}) async {
    final result = await _methodChannel.invokeMapMethod<String, dynamic>('capturePhoto', {
      'folder': folder,
      'prefix': prefix,
    });
    final dets = ((result?['detections'] as List?) ?? []).map((e) => DetectionResult.fromMap(e as Map)).toList();
    return CaptureResult(
      cleanPath: result?['cleanPath'] as String? ?? '',
      annotatedPath: result?['annotatedPath'] as String? ?? '',
      detections: dets,
    );
  }

  /// Stream of detection results from native camera.
  Stream<CameraDetectionEvent> get detectionStream {
    return _eventChannel.receiveBroadcastStream().map((event) {
      final map = event as Map;
      final list = (map['detections'] as List?) ?? [];
      final dets = list.map((e) => DetectionResult.fromMap(e as Map)).toList();
      return CameraDetectionEvent(
        detections: dets,
        imageWidth: (map['imageWidth'] as int?) ?? 0,
        imageHeight: (map['imageHeight'] as int?) ?? 0,
        inferenceMs: (map['inferenceMs'] as int?) ?? 0,
      );
    });
  }

  /// Release native resources.
  Future<void> dispose() {
    return PaddleDetectionPlatform.instance.dispose();
  }
}
