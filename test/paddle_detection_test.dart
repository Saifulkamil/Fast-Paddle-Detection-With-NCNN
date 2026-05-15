// import 'package:flutter_test/flutter_test.dart';
// import 'package:paddle_detection/paddle_detection.dart';
// import 'package:paddle_detection/paddle_detection_platform_interface.dart';
// import 'package:paddle_detection/paddle_detection_method_channel.dart';
// import 'package:plugin_platform_interface/plugin_platform_interface.dart';

// class MockPaddleDetectionPlatform
//     with MockPlatformInterfaceMixin
//     implements PaddleDetectionPlatform {

//   @override
//   Future<String?> getPlatformVersion() => Future.value('42');
// }

// void main() {
//   final PaddleDetectionPlatform initialPlatform = PaddleDetectionPlatform.instance;

//   test('$MethodChannelPaddleDetection is the default instance', () {
//     expect(initialPlatform, isInstanceOf<MethodChannelPaddleDetection>());
//   });

//   test('getPlatformVersion', () async {
//     PaddleDetection paddleDetectionPlugin = PaddleDetection();
//     MockPaddleDetectionPlatform fakePlatform = MockPaddleDetectionPlatform();
//     PaddleDetectionPlatform.instance = fakePlatform;

//     expect(await paddleDetectionPlugin.getPlatformVersion(), '42');
//   });
// }
