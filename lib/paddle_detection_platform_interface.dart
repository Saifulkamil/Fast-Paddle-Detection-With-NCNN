import 'dart:typed_data';

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'paddle_detection_method_channel.dart';
import 'src/detection_result.dart';

abstract class PaddleDetectionPlatform extends PlatformInterface {
  /// Constructs a PaddleDetectionPlatform.
  PaddleDetectionPlatform() : super(token: _token);

  static final Object _token = Object();

  static PaddleDetectionPlatform _instance = MethodChannelPaddleDetection();

  /// The default instance of [PaddleDetectionPlatform] to use.
  ///
  /// Defaults to [MethodChannelPaddleDetection].
  static PaddleDetectionPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [PaddleDetectionPlatform] when
  /// they register themselves.
  static set instance(PaddleDetectionPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  /// Load a PicoDet model from Android assets.
  ///
  /// [paramName] - Name of the .param file in assets (e.g. "picodet.ncnn.param")
  /// [binName] - Name of the .bin file in assets (e.g. "picodet.ncnn.bin")
  /// [numClass] - Number of detection classes in the model
  /// [sizeId] - Input size: 0=320, 1=400, 2=480, 3=560, 4=640
  /// [cpuGpu] - Compute backend: 0=CPU, 1=GPU(Vulkan), 2=GPU(Turnip)
  Future<bool> loadModel({
    required String paramName,
    required String binName,
    int numClass = 1,
    int sizeId = 0,
    int cpuGpu = 0,
  }) {
    throw UnimplementedError('loadModel() has not been implemented.');
  }

  /// Load a PicoDet model from absolute file paths (e.g., user-picked files).
  Future<bool> loadModelFromFile({
    required String paramPath,
    required String binPath,
    int numClass = 1,
    int sizeId = 0,
    int cpuGpu = 0,
  }) {
    throw UnimplementedError('loadModelFromFile() has not been implemented.');
  }

  /// Run detection on an image file.
  Future<List<DetectionResult>> detect({required String imagePath}) {
    throw UnimplementedError('detect() has not been implemented.');
  }

  /// Run detection on raw RGBA pixel bytes (e.g., from camera frame).
  Future<List<DetectionResult>> detectFromBytes({required Uint8List data, required int width, required int height}) {
    throw UnimplementedError('detectFromBytes() has not been implemented.');
  }

  /// Update detection thresholds.
  Future<void> setThreshold({double probThreshold = 0.5, double nmsThreshold = 0.45}) {
    throw UnimplementedError('setThreshold() has not been implemented.');
  }

  /// Release native resources.
  Future<void> dispose() {
    throw UnimplementedError('dispose() has not been implemented.');
  }
}
