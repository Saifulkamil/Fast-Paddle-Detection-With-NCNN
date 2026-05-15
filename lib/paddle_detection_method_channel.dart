import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'paddle_detection_platform_interface.dart';
import 'src/detection_result.dart';

/// An implementation of [PaddleDetectionPlatform] that uses method channels.
class MethodChannelPaddleDetection extends PaddleDetectionPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('paddle_detection');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }

  @override
  Future<bool> loadModel({
    required String paramName,
    required String binName,
    int numClass = 1,
    int sizeId = 0,
    int cpuGpu = 0,
  }) async {
    final result = await methodChannel.invokeMethod<bool>('loadModel', {
      'paramName': paramName,
      'binName': binName,
      'numClass': numClass,
      'sizeId': sizeId,
      'cpuGpu': cpuGpu,
    });
    return result ?? false;
  }

  @override
  Future<bool> loadModelFromFile({
    required String paramPath,
    required String binPath,
    int numClass = 1,
    int sizeId = 0,
    int cpuGpu = 0,
  }) async {
    final result = await methodChannel.invokeMethod<bool>('loadModelFromFile', {
      'paramPath': paramPath,
      'binPath': binPath,
      'numClass': numClass,
      'sizeId': sizeId,
      'cpuGpu': cpuGpu,
    });
    return result ?? false;
  }

  @override
  Future<List<DetectionResult>> detect({required String imagePath}) async {
    final result = await methodChannel.invokeMethod<List<dynamic>>('detect', {'imagePath': imagePath});
    if (result == null) return [];
    return result.map((e) => DetectionResult.fromMap(e as Map<dynamic, dynamic>)).toList();
  }

  @override
  Future<List<DetectionResult>> detectFromBytes({
    required Uint8List data,
    required int width,
    required int height,
  }) async {
    final result = await methodChannel.invokeMethod<List<dynamic>>('detectFromBytes', {
      'data': data,
      'width': width,
      'height': height,
    });
    if (result == null) return [];
    return result.map((e) => DetectionResult.fromMap(e as Map<dynamic, dynamic>)).toList();
  }

  @override
  Future<void> setThreshold({double probThreshold = 0.5, double nmsThreshold = 0.45}) async {
    await methodChannel.invokeMethod('setThreshold', {'probThreshold': probThreshold, 'nmsThreshold': nmsThreshold});
  }

  @override
  Future<void> dispose() async {
    await methodChannel.invokeMethod('dispose');
  }
}
