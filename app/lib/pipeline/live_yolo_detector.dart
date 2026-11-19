import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';

const _tracktLabels = {'target_standing', 'target_fallen', 'rc_car'};
const _liveModelAsset = 'assets/models/live_detect.tflite';
const _fallbackModelAsset = 'assets/models/detect.tflite';

enum _DetectorFormat { ssdPostProcess, yoloRaw }

class LiveDetection {
  final String label;
  final double score;
  final Rect rect;

  const LiveDetection({
    required this.label,
    required this.score,
    required this.rect,
  });

  bool get isCar => label == 'rc_car';
  bool get isTarget => label == 'target_standing' || label == 'target_fallen';

  String get displayLabel => switch (label) {
    'rc_car' => 'RC car',
    'target_standing' => 'target',
    'target_fallen' => 'fallen',
    _ => label,
  };

  LiveDetection copyWith({String? label, double? score, Rect? rect}) {
    return LiveDetection(
      label: label ?? this.label,
      score: score ?? this.score,
      rect: rect ?? this.rect,
    );
  }
}

class LiveYoloDetector {
  static LiveYoloDetector? _shared;
  static Future<LiveYoloDetector?>? _loading;

  final Interpreter _interpreter;
  final List<String> _labels;
  final int _inputW;
  final int _inputH;
  final int _maxDetections;
  final _DetectorFormat _format;
  final TensorType _inputType;
  final List<int> _outputShape;
  final String backendName;
  Float32List? _floatInputBuffer;
  Float32List? _outputBuffer;

  LiveYoloDetector._(
    this._interpreter,
    this._labels,
    this._inputW,
    this._inputH,
    this._maxDetections,
    this._format,
    this._inputType,
    this._outputShape,
    this.backendName,
  );

  static Future<LiveYoloDetector?> load() async {
    final cached = _shared;
    if (cached != null) return cached;
    final pending = _loading;
    if (pending != null) return pending;
    final future = _loadFresh().then((detector) {
      if (detector != null) _shared = detector;
      return detector;
    });
    _loading = future;
    return future.whenComplete(() => _loading = null);
  }

  static Future<LiveYoloDetector?> _loadFresh() async {
    try {
      final labelsText = await rootBundle.loadString(
        'assets/models/labelmap.txt',
      );
      final labels = labelsText
          .split(RegExp(r'\r?\n'))
          .map((l) => l.trim().toLowerCase())
          .where((l) => l.isNotEmpty)
          .toList(growable: false);
      if (!labels.toSet().contains('rc_car')) return null;

      final (bytes, modelAsset) = await _loadBestLiveModelAsset();
      final model = bytes.buffer.asUint8List(
        bytes.offsetInBytes,
        bytes.lengthInBytes,
      );
      final cpuCount = Platform.isAndroid ? Platform.numberOfProcessors : 4;
      final highThreads = math.min(8, math.max(1, cpuCount)).toInt();
      final configs = <_BackendConfig>[
        if (Platform.isAndroid)
          _BackendConfig(
            '$modelAsset NNAPI ${highThreads}T',
            highThreads,
            true,
          ),
        _BackendConfig('$modelAsset CPU ${highThreads}T', highThreads, false),
        if (highThreads != 4) _BackendConfig('$modelAsset CPU 4T', 4, false),
      ];

      LiveYoloDetector? best;
      var bestMs = 1 << 62;
      for (final config in configs) {
        final detector = _tryCreate(model, labels, config);
        if (detector == null) continue;
        final ms = _benchmark(detector);
        if (ms < bestMs) {
          best?.close();
          best = detector;
          bestMs = ms;
        } else {
          detector.close();
        }
      }
      return best;
    } catch (_) {
      return null;
    }
  }

  static Future<(ByteData, String)> _loadBestLiveModelAsset() async {
    try {
      return (await rootBundle.load(_liveModelAsset), 'live416-fp16');
    } catch (_) {
      return (await rootBundle.load(_fallbackModelAsset), 'full512');
    }
  }

  static LiveYoloDetector? _tryCreate(
    Uint8List model,
    List<String> labels,
    _BackendConfig config,
  ) {
    try {
      final options = InterpreterOptions()..threads = config.threads;
      if (Platform.isAndroid) {
        options.useNnApiForAndroid = config.useNnApi;
      }
      final interpreter = Interpreter.fromBuffer(model, options: options);
      final inputTensor = interpreter.getInputTensors().first;
      final inputShape = inputTensor.shape;
      final outputs = interpreter.getOutputTensors();
      final outputShape = outputs.first.shape;
      final format = outputs.length >= 4
          ? _DetectorFormat.ssdPostProcess
          : _DetectorFormat.yoloRaw;
      final maxDetections = format == _DetectorFormat.ssdPostProcess
          ? (outputShape.length >= 2 ? outputShape[1] : 10)
          : _yoloPredictionCount(outputShape);
      if (inputShape.length != 4 || maxDetections <= 0) {
        interpreter.close();
        return null;
      }
      return LiveYoloDetector._(
        interpreter,
        labels,
        inputShape[2],
        inputShape[1],
        maxDetections,
        format,
        inputTensor.type,
        outputShape,
        config.name,
      );
    } catch (_) {
      return null;
    }
  }

  static int _benchmark(LiveYoloDetector detector) {
    final rgb = Uint8List(detector.inputWidth * detector.inputHeight * 3);
    var bestMs = 1 << 62;
    for (var i = 0; i < 4; i++) {
      final sw = Stopwatch()..start();
      detector.detectRgb(
        rgb,
        width: detector.inputWidth,
        height: detector.inputHeight,
        threshold: 0.90,
      );
      final elapsed = sw.elapsedMilliseconds;
      if (i > 0 && elapsed < bestMs) bestMs = elapsed;
    }
    return bestMs;
  }

  int get inputWidth => _inputW;
  int get inputHeight => _inputH;

  List<LiveDetection> detect(cv.Mat frame, {double threshold = 0.25}) {
    try {
      final resized = cv.resize(frame, (
        _inputW,
        _inputH,
      ), interpolation: cv.INTER_AREA);
      final rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB);
      resized.dispose();
      final detections = detectRgb(
        rgb.data,
        width: _inputW,
        height: _inputH,
        threshold: threshold,
      );
      rgb.dispose();
      return detections;
    } catch (_) {
      return const [];
    }
  }

  List<LiveDetection> detectRgb(
    Uint8List rgb, {
    required int width,
    required int height,
    double threshold = 0.25,
  }) {
    try {
      final modelRgb = width == _inputW && height == _inputH
          ? rgb
          : _resizeRgbNearest(rgb, width, height, _inputW, _inputH);
      final input = _inputType == TensorType.float32
          ? _floatInputBytes(modelRgb)
          : modelRgb;

      if (_format == _DetectorFormat.ssdPostProcess) {
        return _detectSsd(input, threshold);
      }
      final output = _rawOutputBuffer();
      _interpreter.run(input, output.buffer);
      return _nms(_decodeYoloOutputFlat(output, threshold), 0.45);
    } catch (_) {
      return const [];
    }
  }

  ByteBuffer _floatInputBytes(Uint8List rgb) {
    final input = _floatInputBuffer ??= Float32List(_inputW * _inputH * 3);
    const scale = 1.0 / 255.0;
    for (var i = 0; i < rgb.length; i++) {
      input[i] = rgb[i] * scale;
    }
    return input.buffer;
  }

  Float32List _rawOutputBuffer() {
    final n = _shapeProduct(_outputShape);
    final current = _outputBuffer;
    if (current != null && current.length == n) return current;
    return _outputBuffer = Float32List(n);
  }

  List<LiveDetection> _detectSsd(Object input, double threshold) {
    final boxes = List.generate(
      1,
      (_) => List.generate(_maxDetections, (_) => List<double>.filled(4, 0)),
    );
    final classes = List.generate(
      1,
      (_) => List<double>.filled(_maxDetections, 0),
    );
    final scores = List.generate(
      1,
      (_) => List<double>.filled(_maxDetections, 0),
    );
    final count = List<double>.filled(1, 0);

    _interpreter.runForMultipleInputs(
      [input],
      {0: boxes, 1: classes, 2: scores, 3: count},
    );

    final detections = <LiveDetection>[];
    final n = count.first.round().clamp(0, _maxDetections);
    for (int i = 0; i < n; i++) {
      final score = scores[0][i];
      if (score < threshold) continue;
      final label = _labelForClass(classes[0][i]);
      if (!_tracktLabels.contains(label)) continue;
      final b = boxes[0][i];
      final rect = Rect.fromLTRB(
        b[1].clamp(0.0, 1.0),
        b[0].clamp(0.0, 1.0),
        b[3].clamp(0.0, 1.0),
        b[2].clamp(0.0, 1.0),
      );
      detections.add(LiveDetection(label: label, score: score, rect: rect));
    }
    return detections;
  }

  List<LiveDetection> _decodeYoloOutputFlat(
    Float32List output,
    double threshold,
  ) {
    if (_outputShape.length != 3 || output.isEmpty) return const [];

    final labelCount = _labels.length;
    final channelsFirst =
        _outputShape[1] == labelCount + 4 ||
        (_outputShape[1] < _outputShape[2] &&
            _outputShape[1] <= labelCount + 8);
    final channels = channelsFirst ? _outputShape[1] : _outputShape[2];
    final predictions = channelsFirst ? _outputShape[2] : _outputShape[1];
    if (channels < 5 || predictions <= 0) return const [];

    double valueAt(int pred, int ch) {
      final idx = channelsFirst
          ? ch * predictions + pred
          : pred * channels + ch;
      if (idx < 0 || idx >= output.length) return 0;
      return output[idx];
    }

    final detections = <LiveDetection>[];
    for (int p = 0; p < predictions; p++) {
      if (!channelsFirst && channels == 6) {
        final score = valueAt(p, 4);
        if (score < threshold) continue;
        final label = _labelForClass(valueAt(p, 5));
        if (!_tracktLabels.contains(label)) continue;
        final rect = _rectFromCompactOutput(
          valueAt(p, 0),
          valueAt(p, 1),
          valueAt(p, 2),
          valueAt(p, 3),
          _inputW,
          _inputH,
        );
        if (rect == null) continue;
        detections.add(LiveDetection(label: label, score: score, rect: rect));
        continue;
      }

      var bestScore = 0.0;
      var bestClass = -1;
      final classChannels = math.min(labelCount, channels - 4);
      for (int c = 0; c < classChannels; c++) {
        final score = valueAt(p, 4 + c);
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < threshold || bestClass < 0) continue;
      final label = _labelForClass(bestClass.toDouble());
      if (!_tracktLabels.contains(label)) continue;

      var cx = valueAt(p, 0);
      var cy = valueAt(p, 1);
      var w = valueAt(p, 2).abs();
      var h = valueAt(p, 3).abs();
      final normalized = [cx, cy, w, h].every((v) => v >= 0 && v <= 1.5);
      if (!normalized) {
        cx /= _inputW;
        w /= _inputW;
        cy /= _inputH;
        h /= _inputH;
      }
      final rect = Rect.fromLTRB(
        (cx - w / 2).clamp(0.0, 1.0),
        (cy - h / 2).clamp(0.0, 1.0),
        (cx + w / 2).clamp(0.0, 1.0),
        (cy + h / 2).clamp(0.0, 1.0),
      );
      if (rect.width <= 0 || rect.height <= 0) continue;
      detections.add(LiveDetection(label: label, score: bestScore, rect: rect));
    }
    return detections;
  }

  String _labelForClass(double rawClass) {
    final raw = rawClass.round();
    final offset = _labels.isNotEmpty && _labels.first == '???' ? 1 : 0;
    final idx = raw + offset;
    if (idx < 0 || idx >= _labels.length) return '';
    return _labels[idx].toLowerCase();
  }

  void close() => _interpreter.close();
}

class _BackendConfig {
  final String name;
  final int threads;
  final bool useNnApi;

  const _BackendConfig(this.name, this.threads, this.useNnApi);
}

Rect? _rectFromCompactOutput(
  double x1,
  double y1,
  double x2,
  double y2,
  int inputW,
  int inputH,
) {
  final normalized = [x1, y1, x2, y2].every((v) => v >= 0 && v <= 1.5);
  if (!normalized) {
    x1 /= inputW;
    x2 /= inputW;
    y1 /= inputH;
    y2 /= inputH;
  }
  if (x2 <= x1 || y2 <= y1) {
    final cx = x1;
    final cy = y1;
    final w = x2.abs();
    final h = y2.abs();
    x1 = cx - w / 2;
    x2 = cx + w / 2;
    y1 = cy - h / 2;
    y2 = cy + h / 2;
  }
  final rect = Rect.fromLTRB(
    x1.clamp(0.0, 1.0),
    y1.clamp(0.0, 1.0),
    x2.clamp(0.0, 1.0),
    y2.clamp(0.0, 1.0),
  );
  if (rect.width <= 0 || rect.height <= 0) return null;
  return rect;
}

int _shapeProduct(List<int> shape) =>
    shape.fold<int>(1, (acc, v) => acc * math.max(1, v));

int _yoloPredictionCount(List<int> shape) {
  if (shape.length != 3) return 0;
  return math.max(shape[1], shape[2]);
}

Uint8List _resizeRgbNearest(
  Uint8List src,
  int srcW,
  int srcH,
  int dstW,
  int dstH,
) {
  final dst = Uint8List(dstW * dstH * 3);
  for (var y = 0; y < dstH; y++) {
    final sy = (y * srcH / dstH).floor().clamp(0, srcH - 1);
    for (var x = 0; x < dstW; x++) {
      final sx = (x * srcW / dstW).floor().clamp(0, srcW - 1);
      final si = (sy * srcW + sx) * 3;
      final di = (y * dstW + x) * 3;
      dst[di] = src[si];
      dst[di + 1] = src[si + 1];
      dst[di + 2] = src[si + 2];
    }
  }
  return dst;
}

List<LiveDetection> _nms(List<LiveDetection> detections, double iouThreshold) {
  final sorted = detections.toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  final kept = <LiveDetection>[];
  for (final det in sorted) {
    final overlaps = kept.any(
      (k) => k.label == det.label && _rectIoU(k.rect, det.rect) > iouThreshold,
    );
    if (!overlaps) kept.add(det);
  }
  return kept;
}

double _rectIoU(Rect a, Rect b) {
  final left = math.max(a.left, b.left);
  final top = math.max(a.top, b.top);
  final right = math.min(a.right, b.right);
  final bottom = math.min(a.bottom, b.bottom);
  if (right <= left || bottom <= top) return 0.0;
  final inter = (right - left) * (bottom - top);
  final union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0.0 : inter / union;
}
