/// Detection result from PicoDet model.
class DetectionResult {
  /// Class label index (0-based).
  final int label;

  /// Confidence probability (0.0 - 1.0).
  final double probability;

  /// Bounding box x coordinate (absolute pixels).
  final double x;

  /// Bounding box y coordinate (absolute pixels).
  final double y;

  /// Bounding box width (absolute pixels).
  final double width;

  /// Bounding box height (absolute pixels).
  final double height;

  const DetectionResult({
    required this.label,
    required this.probability,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory DetectionResult.fromMap(Map<dynamic, dynamic> map) {
    return DetectionResult(
      label: (map['label'] as num).toInt(),
      probability: (map['prob'] as num).toDouble(),
      x: (map['x'] as num).toDouble(),
      y: (map['y'] as num).toDouble(),
      width: (map['width'] as num).toDouble(),
      height: (map['height'] as num).toDouble(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'label': label,
      'prob': probability,
      'x': x,
      'y': y,
      'width': width,
      'height': height,
    };
  }

  @override
  String toString() =>
      'DetectionResult(label: $label, prob: ${probability.toStringAsFixed(3)}, '
      'rect: [${x.toStringAsFixed(1)}, ${y.toStringAsFixed(1)}, '
      '${width.toStringAsFixed(1)}, ${height.toStringAsFixed(1)}])';
}
