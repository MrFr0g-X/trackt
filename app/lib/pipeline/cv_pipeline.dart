// ignore_for_file: avoid_print, unused_element
import 'dart:isolate';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/pipeline_result.dart';
import 'preprocess.dart';

const _minTargetArea = 130.0;
const _minMoverArea = 650.0;
const _minFillRatio = 0.18;
const _maxJumpFrac = 0.16;
const _minSeedAreaFraction = 0.0018;

const _fallRatioFactor = 0.55;
const _fallAreaFactor = 0.40;
const _fallConfirmFrames = 7;
const _minShapeFallAgeSec = 0.45;
const _minAreaFallAgeSec = 1.25;
const _minVanishFallAgeSec = 0.45;
const _disappearThresh = 8;
const _impactBurnFrames = 240;
const _impactBurnPad = 32;
const _maxMoverAreaJump = 3.5;
const _moverRetireTtl = 240;
const _moverReIdAreaTol = 0.55;
const _minTrackedFrames = 3;
const _maxTracksPerLabel = 6;
const _maxMoverTracks = 1;
const _minMoverPathPoints = 2;

const _discoverFrames = 60;
const _targetSeedFrames = 12;
const _seedMinHitRatio = 0.18;
const _seedMinHits = 2;
const _discoverWidth = 192;
const _discoverHeight = 108;
const _hueWindow = 12;
const _hueMergeDistance = 18;
const _maxDiscoveredColors = 8;
const _minSaturation = 60;
const _minValue = 50;
const _maxValue = 240;
const _baseDetectRatio = 0.06;

const _sampleEveryFrames = 30;
const _semanticScanEvery = 1;
const _semanticCupThreshold = 0.45;
const _semanticCarThreshold = 0.12;
const _semanticMoverJumpMultiplier = 99.0;
const _customDetectorLabels = {'target_standing', 'target_fallen', 'rc_car'};
const _liveSemanticModelAsset = 'assets/models/live_detect.tflite';
const _fallbackSemanticModelAsset = 'assets/models/detect.tflite';

class _ColorSpec {
  final List<List<int>> ranges;
  final int displayBgrPacked;
  _ColorSpec(this.ranges, this.displayBgrPacked);
}

Map<String, _ColorSpec> _assignmentTargetSpecs() => {
  'red': _ColorSpec([
    [0, 14, 65, 45],
    [164, 179, 65, 45],
  ], 0xFFEF4444),
  'blue': _ColorSpec([
    [88, 133, 55, 35],
  ], 0xFF2563EB),
  'yellow': _ColorSpec([
    [15, 42, 90, 70],
  ], 0xFFFACC15),
};

Map<String, List<List<int>>> _targetRangeMap(Map<String, _ColorSpec> specs) => {
  for (final entry in specs.entries)
    entry.key: entry.value.ranges.map((r) => List<int>.from(r)).toList(),
};

const _autoDiscoverFrames = 60;
const _autoDiscoverWidth = 192;
const _autoDiscoverHeight = 108;
const _autoHueWindow = 12;
const _autoHueMergeDist = 18;
const _autoMaxColors = 8;
const _autoMinSaturation = 60;
const _autoMinValue = 50;
const _autoMaxValue = 240;

Map<String, _ColorSpec> _autoDiscoverSpecs(
  String inputPath, {
  void Function(double)? onProgress,
}) {
  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) return const {};

  final hueHist = List<int>.filled(180, 0);
  final samples = <List<int>>[];
  int scanned = 0;

  for (int i = 0; i < _autoDiscoverFrames; i++) {
    final (ok, raw) = cap.read();
    if (!ok || raw.isEmpty) {
      raw.dispose();
      break;
    }
    scanned++;
    final small = cv.resize(raw, (
      _autoDiscoverWidth,
      _autoDiscoverHeight,
    ), interpolation: cv.INTER_AREA);
    raw.dispose();
    final hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV);
    small.dispose();

    final h = cv.extractChannel(hsv, 0);
    final s = cv.extractChannel(hsv, 1);
    final v = cv.extractChannel(hsv, 2);
    hsv.dispose();

    final hData = h.data;
    final sData = s.data;
    final vData = v.data;
    for (int p = 0; p < hData.length; p++) {
      final ss = sData[p], vv = vData[p];
      if (ss < _autoMinSaturation || vv < _autoMinValue || vv > _autoMaxValue) {
        continue;
      }
      hueHist[hData[p]]++;
      if (samples.length < 5000) samples.add([hData[p], ss, vv]);
    }
    h.dispose();
    s.dispose();
    v.dispose();
    onProgress?.call(i / _autoDiscoverFrames);
  }
  cap.release();
  if (scanned == 0) return const {};

  final smoothed = List<double>.filled(180, 0);
  for (int i = 0; i < 180; i++) {
    double sum = 0;
    int cnt = 0;
    for (int j = -3; j <= 3; j++) {
      final idx = (i + j + 180) % 180;
      sum += hueHist[idx];
      cnt++;
    }
    smoothed[i] = sum / cnt;
  }

  final maxVal = smoothed.fold<double>(0, math.max);
  if (maxVal <= 0) return const {};
  final minHeight = maxVal * 0.05;

  final peaks = <int>[];
  for (int i = 0; i < 180; i++) {
    final left = smoothed[(i - 1 + 180) % 180];
    final right = smoothed[(i + 1) % 180];
    if (smoothed[i] > left && smoothed[i] > right && smoothed[i] >= minHeight) {
      peaks.add(i);
    }
  }
  peaks.sort((a, b) => smoothed[b].compareTo(smoothed[a]));
  final keepPeaks = <int>[];
  for (final pk in peaks) {
    bool tooClose = false;
    for (final k in keepPeaks) {
      var d = (pk - k).abs();
      if (d > 90) d = 180 - d;
      if (d < _autoHueMergeDist) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) keepPeaks.add(pk);
    if (keepPeaks.length >= _autoMaxColors) break;
  }

  final specs = <String, _ColorSpec>{};
  final usedNames = <String>{};
  for (final hue in keepPeaks) {
    int sSum = 0, vSum = 0, n = 0;
    for (final s in samples) {
      var d = (s[0] - hue).abs();
      if (d > 90) d = 180 - d;
      if (d <= _autoHueWindow) {
        sSum += s[1];
        vSum += s[2];
        n++;
      }
    }
    final sAvg = n > 0 ? (sSum / n).round() : 200;
    final vAvg = n > 0 ? (vSum / n).round() : 200;

    final ranges = <List<int>>[];
    final lo = hue - _autoHueWindow;
    final hi = hue + _autoHueWindow;
    final sLo = math.max(_autoMinSaturation, sAvg - 80);
    final vLo = math.max(_autoMinValue, vAvg - 80);
    if (lo < 0) {
      ranges.add([0, hi, sLo, vLo]);
      ranges.add([180 + lo, 179, sLo, vLo]);
    } else if (hi > 179) {
      ranges.add([lo, 179, sLo, vLo]);
      ranges.add([0, hi - 180, sLo, vLo]);
    } else {
      ranges.add([lo, hi, sLo, vLo]);
    }

    final rgb = _hsvToRgb(hue, sAvg, vAvg);
    final packed = (0xFF << 24) | (rgb.$1 << 16) | (rgb.$2 << 8) | rgb.$3;

    var name = _nameForHue(hue);
    var key = name;
    int suffix = 2;
    while (usedNames.contains(key)) {
      key = '$name $suffix';
      suffix++;
    }
    usedNames.add(key);

    specs[key] = _ColorSpec(ranges, packed);
  }
  return specs;
}

class _Track {
  final String id;
  final String label;
  final int firstFrame;
  final int? targetNumber;

  cv.Rect box;
  double area;
  double ratio;
  double maxArea = 0;
  double confidence = 0;

  int trackedFrames = 0;
  int missingFrames = 0;
  int fallCounter = 0;

  double initialArea = 0;
  double initialRatio = 0;
  double initialCx = 0;
  double initialCy = 0;
  double initialWidth = 0;
  double initialHeight = 0;
  cv.Rect? baselineBox;
  bool baselineSet = false;
  int? fallFrame;
  int lastSeenFrame = 0;

  cv.Rect? fallBox;
  String? fallSignal;

  final List<List<(int, int)>> pathSegments = [<(int, int)>[]];
  double smCx = -1, smCy = -1;
  double lastVx = 0, lastVy = 0;

  List<(int, int)> get path => [for (final seg in pathSegments) ...seg];

  _Track({
    required this.id,
    required this.label,
    required this.box,
    required this.area,
    required this.ratio,
    required this.firstFrame,
    this.targetNumber,
  }) {
    maxArea = area;
  }

  double get cx => box.x + box.width / 2.0;
  double get cy => box.y + box.height / 2.0;
}

class _RetiredMover {
  final String id;
  final cv.Rect lastBox;
  final double lastArea;
  final double lastCx;
  final double lastCy;
  final double lastVx;
  final double lastVy;
  final int retiredAtFrame;
  final List<List<(int, int)>> pathSegments;
  final double bestConfidence;

  const _RetiredMover({
    required this.id,
    required this.lastBox,
    required this.lastArea,
    required this.lastCx,
    required this.lastCy,
    required this.lastVx,
    required this.lastVy,
    required this.retiredAtFrame,
    required this.pathSegments,
    required this.bestConfidence,
  });
}

class _Seed {
  final String label;
  cv.Rect box;
  double area;
  int hits = 1;
  int lastFrame;

  _Seed({
    required this.label,
    required this.box,
    required this.area,
    required this.lastFrame,
  });

  double get cx => box.x + box.width / 2.0;
  double get cy => box.y + box.height / 2.0;
  double get score => hits * math.sqrt(area.clamp(1.0, double.infinity));

  void update(_Blob blob, int frameIdx) {
    box = blob.box;
    area = area * 0.75 + blob.area * 0.25;
    hits++;
    lastFrame = frameIdx;
  }
}

class _WriterHandle {
  final cv.VideoWriter? writer;
  final String outputPath;
  final String codec;

  const _WriterHandle({
    required this.writer,
    required this.outputPath,
    required this.codec,
  });

  bool get isOpen => writer != null;
}

class _SemanticAssets {
  final TransferableTypedData? model;
  final List<String> labels;
  final String modelName;

  const _SemanticAssets(this.model, this.labels, this.modelName);
}

class _SemanticDetection {
  final String label;
  final double score;
  final cv.Rect box;

  const _SemanticDetection(this.label, this.score, this.box);
}

bool _passesSemanticRuntimeThreshold(_SemanticDetection detection) {
  if (detection.label == 'rc_car') {
    return detection.score >= _semanticCarThreshold;
  }
  return detection.score >= _semanticCupThreshold;
}

Future<PipelineResult> runPipeline(
  String inputPath,
  String outputPath, {
  PipelineOptions options = const PipelineOptions(),
  void Function(PipelineProgress)? onProgress,
  void Function(LogEvent)? onLog,
  void Function(CountSample)? onCount,
}) async {
  final receive = ReceivePort();
  Map<String, dynamic>? finalMap;
  String? errorMsg;
  final semanticAssets = options.semanticAssist
      ? await _loadSemanticAssets()
      : const _SemanticAssets(null, [], '');

  await Isolate.spawn<List<dynamic>>(_isolateEntry, [
    receive.sendPort,
    inputPath,
    outputPath,
    options.confidenceThreshold,
    options.downscaleHeight,
    options.maxTargets,
    options.minDisplayConfidence,
    options.debugMode,
    options.expectedTargetCount,
    semanticAssets.model,
    semanticAssets.labels,
    semanticAssets.modelName,
    options.targetMode.name,
    options.dualStreamPreprocessing,
    options.illuminationNormalization,
    options.frameStep,
    options.writeAnnotatedVideo,
    options.earlyStopOnComplete,
  ]);

  await for (final raw in receive) {
    if (raw is! Map) continue;
    switch (raw['type']) {
      case 'progress':
        onProgress?.call(PipelineProgress.fromMap(raw));
        break;
      case 'log':
        onLog?.call(LogEvent.fromMap(raw));
        break;
      case 'count':
        onCount?.call(CountSample.fromMap(raw));
        break;
      case 'done':
        finalMap = Map<String, dynamic>.from(raw['result'] as Map);
        receive.close();
        break;
      case 'error':
        errorMsg = raw['message'] as String;
        receive.close();
        break;
    }
  }

  if (errorMsg != null) throw Exception(errorMsg);
  if (finalMap == null) throw Exception('Pipeline produced no result');
  return PipelineResult.fromMap(finalMap, options);
}

Future<_SemanticAssets> _loadSemanticAssets() async {
  try {
    final labelsText = await rootBundle.loadString(
      'assets/models/labelmap.txt',
    );
    final labels = labelsText
        .split(RegExp(r'\r?\n'))
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();

    if (!_hasCustomDetectorLabels(labels)) {
      return _SemanticAssets(null, labels, '');
    }

    ByteData bytes;
    String modelName;
    try {
      bytes = await rootBundle.load(_liveSemanticModelAsset);
      modelName = 'live_detect.tflite';
    } catch (_) {
      bytes = await rootBundle.load(_fallbackSemanticModelAsset);
      modelName = 'detect.tflite';
    }
    final view = bytes.buffer.asUint8List(
      bytes.offsetInBytes,
      bytes.lengthInBytes,
    );
    return _SemanticAssets(
      TransferableTypedData.fromList([view]),
      labels,
      modelName,
    );
  } catch (_) {
    return const _SemanticAssets(null, [], '');
  }
}

bool _hasCustomDetectorLabels(List<String> labels) {
  final normalized = labels.map((l) => l.trim().toLowerCase()).toSet();
  return normalized.contains('rc_car');
}

void _isolateEntry(List<dynamic> args) {
  final SendPort send = args[0] as SendPort;
  final String inputPath = args[1] as String;
  final String outputPath = args[2] as String;
  final double confThreshold = args[3] as double;
  final int? downscaleHeight = args[4] as int?;
  final int maxTargets = args[5] as int;
  final double minDisplayConfidence = args[6] as double;
  final bool debugMode = args[7] as bool;
  final int? expectedTargetCount = args[8] as int?;
  final TransferableTypedData? semanticModel =
      args[9] as TransferableTypedData?;
  final List<String> semanticLabels = List<String>.from(args[10] as List);
  final String semanticModelName = args.length > 11 ? args[11] as String : '';
  final String targetModeName = args.length > 12
      ? args[12] as String
      : TargetMode.controlled.name;
  final bool dualStreamPreprocessing = args.length > 13
      ? args[13] as bool
      : true;
  final bool illuminationNormalization = args.length > 14
      ? args[14] as bool
      : false;
  final int requestedFrameStep = args.length > 15 ? args[15] as int : 2;
  final bool writeAnnotatedVideo = args.length > 16 ? args[16] as bool : true;
  final bool earlyStopOnComplete = args.length > 17 ? args[17] as bool : false;

  try {
    _syncPipeline(
      send: send,
      inputPath: inputPath,
      outputPath: outputPath,
      confThreshold: confThreshold,
      downscaleHeight: downscaleHeight,
      maxTargets: maxTargets,
      minDisplayConfidence: minDisplayConfidence,
      debugMode: debugMode,
      expectedTargetCount: expectedTargetCount,
      semanticModelBytes: semanticModel?.materialize().asUint8List(),
      semanticLabels: semanticLabels,
      semanticModelName: semanticModelName,
      targetMode: TargetMode.values.firstWhere(
        (m) => m.name == targetModeName,
        orElse: () => TargetMode.controlled,
      ),
      dualStreamPreprocessing: dualStreamPreprocessing,
      illuminationNormalization: illuminationNormalization,
      requestedFrameStep: requestedFrameStep,
      writeAnnotatedVideo: writeAnnotatedVideo,
      earlyStopOnComplete: earlyStopOnComplete,
    );
  } catch (e, st) {
    send.send({'type': 'error', 'message': '$e\n$st'});
  }
}

void _syncPipeline({
  required SendPort send,
  required String inputPath,
  required String outputPath,
  required double confThreshold,
  required int? downscaleHeight,
  required int maxTargets,
  required double minDisplayConfidence,
  required bool debugMode,
  required int? expectedTargetCount,
  required Uint8List? semanticModelBytes,
  required List<String> semanticLabels,
  required String semanticModelName,
  TargetMode targetMode = TargetMode.controlled,
  bool dualStreamPreprocessing = true,
  bool illuminationNormalization = false,
  int requestedFrameStep = 2,
  bool writeAnnotatedVideo = true,
  bool earlyStopOnComplete = false,
}) {
  final t0 = DateTime.now();
  final eventLog = <Map<String, dynamic>>[];
  final countTimeline = <Map<String, dynamic>>[];

  void emitProgress(PipelineStage stage, double p, String msg) {
    send.send({
      'type': 'progress',
      'stage': stage.name,
      'progress': p,
      'message': msg,
    });
  }

  void emitLog(int frame, double t, String msg, String level, {String? color}) {
    final entry = {
      'frame': frame,
      't': t,
      'msg': msg,
      'level': level,
      if (color != null) 'color': color,
    };
    eventLog.add(entry);
    send.send({'type': 'log', ...entry});
  }

  void emitCount(double t, Map<String, int> counts) {
    countTimeline.add({'t': t, 'counts': counts});
    send.send({'type': 'count', 't': t, 'counts': counts});
  }

  emitProgress(PipelineStage.load, 0.0, 'Opening video...');

  final probeCap = cv.VideoCapture.fromFile(inputPath);
  if (!probeCap.isOpened) throw Exception('Cannot open: $inputPath');
  final origH = probeCap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
  final origW = probeCap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
  final fps = probeCap.get(cv.CAP_PROP_FPS).clamp(1.0, 120.0);
  probeCap.release();

  double scale = 1.0;
  if (downscaleHeight != null && origH > downscaleHeight) {
    scale = downscaleHeight / origH;
  }
  final W = _evenDimension((origW * scale).round());
  final H = _evenDimension((origH * scale).round());
  final downsampling = (scale - 1.0).abs() > 0.001;

  emitLog(
    0,
    0,
    'Loaded ${origW}x$origH @ ${fps.toStringAsFixed(1)} fps'
        '${downsampling ? ' → working at ${W}x$H' : ''}',
    'info',
  );
  emitProgress(PipelineStage.load, 1.0, 'Loaded');

  emitProgress(PipelineStage.detect, 0.0, 'Loading target colour model...');
  Map<String, _ColorSpec> activeSpecs;
  if (targetMode == TargetMode.auto) {
    activeSpecs = _autoDiscoverSpecs(
      inputPath,
      onProgress: (p) => emitProgress(
        PipelineStage.detect,
        p,
        'Discovering target colours...',
      ),
    );
    if (activeSpecs.isEmpty) {
      emitLog(
        0,
        0,
        'Auto discovery found no colour clusters — falling back to controlled model',
        'warn',
      );
      activeSpecs = _assignmentTargetSpecs();
    } else {
      emitLog(
        0,
        0,
        'Auto-discovered ${activeSpecs.length} colour cluster(s): ${activeSpecs.keys.join(", ")}',
        'success',
      );
    }
  } else {
    activeSpecs = _assignmentTargetSpecs();
    emitLog(
      0,
      0,
      'Using controlled target model: 2 red, 2 blue, 2 yellow',
      'success',
    );
  }
  emitProgress(PipelineStage.detect, 1.0, 'Loaded target model');

  final targetRanges = _targetRangeMap(activeSpecs);
  final preprocessProfile = dualStreamPreprocessing
      ? calibrateScenePreprocess(
          inputPath,
          scale: scale,
          width: W,
          height: H,
          targetRanges: targetRanges,
        )
      : const ScenePreprocessProfile.disabled();
  if (preprocessProfile.enabled) {
    emitLog(
      0,
      0,
      'Dual-stream preprocessing enabled: car-like HSV≈'
          '${preprocessProfile.avgCarHue.toStringAsFixed(0)}/'
          '${preprocessProfile.avgCarSaturation.toStringAsFixed(0)}/'
          '${preprocessProfile.avgCarValue.toStringAsFixed(0)}, '
          'dark cutoff=${preprocessProfile.darkValueCutoff}',
      'success',
    );
  } else if (dualStreamPreprocessing) {
    emitLog(
      0,
      0,
      'Dual-stream preprocessing enabled with default car profile',
      'info',
    );
  }

  final semanticDetector = _SemanticDetector.tryCreate(
    semanticModelBytes,
    semanticLabels,
  );
  if (semanticDetector != null) {
    emitLog(
      0,
      0,
      'Loaded custom TFLite detector assist'
          '${semanticModelName.isNotEmpty ? " ($semanticModelName)" : ""}',
      'success',
    );
  } else if (_hasCustomDetectorLabels(semanticLabels)) {
    emitLog(
      0,
      0,
      'Custom detector labels found, but TFLite model could not be opened; using CV fallback',
      'warn',
    );
  } else {
    emitLog(0, 0, 'No custom detector bundled yet; using CV fallback', 'info');
  }

  final seeds = _seedInitialTargets(
    inputPath,
    scale,
    W,
    H,
    activeSpecs,
    maxTargets: maxTargets,
    expectedTargetCount: expectedTargetCount,
    semanticDetector: semanticDetector,
    targetRanges: targetRanges,
    targetMode: targetMode,
    dualStreamPreprocessing: dualStreamPreprocessing,
    illuminationNormalization: illuminationNormalization,
    onProgress: (p) => emitProgress(
      PipelineStage.classify,
      p * 0.25,
      'Locking stable target objects',
    ),
  );
  emitLog(0, 0, 'Locked ${seeds.length} stable target object(s)', 'success');

  emitProgress(PipelineStage.classify, 0.0, 'Tracking objects...');

  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) throw Exception('Cannot reopen: $inputPath');
  final totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();
  final frameStep = requestedFrameStep.clamp(1, 12).toInt();
  final outputFps = fps / frameStep;

  final writerHandle = writeAnnotatedVideo
      ? _openBestWriter(outputPath, outputFps, W, H)
      : const _WriterHandle(writer: null, outputPath: '', codec: '');
  final writer = writerHandle.writer;
  if (!writeAnnotatedVideo) {
    emitLog(
      0,
      0,
      'Fast analysis mode: annotated video writing disabled',
      'info',
    );
  } else if (writer == null) {
    emitLog(
      0,
      0,
      'Annotated video encoding is unavailable on this Android build; analysis will still run',
      'warn',
    );
  } else if (writerHandle.outputPath != outputPath ||
      writerHandle.codec != 'mp4v') {
    emitLog(
      0,
      0,
      'Using ${writerHandle.codec} writer: ${writerHandle.outputPath}',
      'info',
    );
  }

  final bg = cv.createBackgroundSubtractorMOG2(
    history: 500,
    varThreshold: 50,
    detectShadows: false,
  );
  final k7 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7));
  final k11 = cv.getStructuringElement(cv.MORPH_RECT, (11, 11));
  cv.Mat? previousMotionGray;

  final tracks = <String, _Track>{};
  final retiredMovers = <_RetiredMover>[];
  final perLabelCounter = <String, int>{};
  String mintTrackId(String label) {
    perLabelCounter[label] = (perLabelCounter[label] ?? 0) + 1;
    return '$label#${perLabelCounter[label]}';
  }

  final orderedSeeds = seeds.toList()
    ..sort((a, b) {
      final ax = a.box.x + a.box.width / 2.0;
      final bx = b.box.x + b.box.width / 2.0;
      final x = ax.compareTo(bx);
      if (x != 0) return x;
      final ay = a.box.y + a.box.height / 2.0;
      final by = b.box.y + b.box.height / 2.0;
      return ay.compareTo(by);
    });

  for (int i = 0; i < orderedSeeds.length; i++) {
    final seed = orderedSeeds[i];
    final id = mintTrackId(seed.label);
    final ratio = seed.box.width > 0
        ? seed.box.height / seed.box.width.toDouble()
        : 0.0;
    final track = _Track(
      id: id,
      label: seed.label,
      box: seed.box,
      area: seed.area,
      ratio: ratio,
      firstFrame: 0,
      targetNumber: i + 1,
    );
    track.confidence = 0.75;
    track.lastSeenFrame = 0;
    tracks[id] = track;
  }

  final maxJumpPx = _maxJumpFrac * math.max(W, H);

  int frameIdx = 0;
  int processedFrameCount = 0;
  int? completeAtFrame;
  final int totalProcessFrames = totalFrames > 0
      ? (totalFrames / frameStep).ceil().clamp(1, 1 << 30).toInt()
      : 0;
  var lastAnnotateProgressAt = DateTime.fromMillisecondsSinceEpoch(0);
  emitProgress(
    PipelineStage.annotate,
    0.01,
    totalProcessFrames > 0
        ? 'Preparing annotation: 0 / $totalProcessFrames frames'
        : 'Preparing annotation...',
  );

  while (true) {
    final (ok, rawFrame) = cap.read();
    if (!ok || rawFrame.isEmpty) {
      rawFrame.dispose();
      break;
    }

    if (frameIdx % frameStep != 0) {
      rawFrame.dispose();
      frameIdx++;
      continue;
    }

    final nowBeforeFrame = DateTime.now();
    if (nowBeforeFrame.difference(lastAnnotateProgressAt).inMilliseconds >=
        900) {
      final p = totalProcessFrames > 0
          ? (processedFrameCount / totalProcessFrames).clamp(0.01, 0.99)
          : 0.01;
      emitProgress(
        PipelineStage.annotate,
        p,
        totalProcessFrames > 0
            ? 'Processing frame ${processedFrameCount + 1} / $totalProcessFrames'
            : 'Processing frame ${processedFrameCount + 1}',
      );
      lastAnnotateProgressAt = nowBeforeFrame;
    }

    cv.Mat frame;
    if (downsampling) {
      frame = cv.resize(rawFrame, (W, H), interpolation: cv.INTER_AREA);
      rawFrame.dispose();
    } else {
      frame = rawFrame;
    }

    final targetFrame = dualStreamPreprocessing
        ? enhanceTargetsForModel(frame, targetRanges)
        : (illuminationNormalization && targetMode == TargetMode.auto)
        ? applyIlluminationNorm(frame)
        : frame;
    final carFrame = dualStreamPreprocessing
        ? enhanceCarForModel(frame, targetRanges, preprocessProfile)
        : frame;

    List<_SemanticDetection> semanticDetections;
    if (semanticDetector != null && frameIdx % _semanticScanEvery == 0) {
      final enhancedDetections = semanticDetector
          .detect(carFrame, threshold: _semanticCarThreshold)
          .where(_passesSemanticRuntimeThreshold)
          .toList(growable: false);

      if (dualStreamPreprocessing && !identical(carFrame, frame)) {
        final originalDetections = semanticDetector
            .detect(frame, threshold: _semanticCarThreshold)
            .where(_passesSemanticRuntimeThreshold)
            .toList(growable: false);
        semanticDetections = _nmsDetections([
          ...enhancedDetections,
          ...originalDetections,
        ], 0.45);
      } else {
        semanticDetections = enhancedDetections;
      }
    } else {
      semanticDetections = const <_SemanticDetection>[];
    }

    final hsv = cv.cvtColor(targetFrame, cv.COLOR_BGR2HSV);
    final visibleCounts = <String, int>{};

    for (final entry in activeSpecs.entries) {
      final colorKey = entry.key;
      final spec = entry.value;
      final mask = _colorMask(hsv, spec, k7);
      final blobs = _allBlobs(mask, _minTargetArea)
          .expand(_splitMergedBlobIfNeeded)
          .where((b) => _isRuntimeTargetCandidate(b, W, H))
          .toList();
      visibleCounts[colorKey] = blobs.length;

      final existing = tracks.values
          .where((t) => t.label == colorKey && t.fallFrame == null)
          .toList();
      _matchAndUpdate(
        tracks: tracks,
        existing: existing,
        blobs: blobs,
        label: colorKey,
        frameIdx: frameIdx,
        sourceFps: fps,
        effectiveFps: outputFps,
        frameStep: frameStep,
        maxJump: maxJumpPx,
        mintId: mintTrackId,
        emitLog: emitLog,
        canSpawn: false,
      );
    }
    hsv.dispose();

    final fg = bg.apply(carFrame);
    final fgOpen = cv.morphologyEx(fg, cv.MORPH_OPEN, k11);
    final fgClose = cv.morphologyEx(fgOpen, cv.MORPH_CLOSE, k11);
    fg.dispose();
    fgOpen.dispose();

    final motionGray = cv.cvtColor(carFrame, cv.COLOR_BGR2GRAY);

    cv.Mat motionMask;
    if (previousMotionGray == null) {
      motionMask = fgClose;
    } else {
      final diff = cv.absDiff(motionGray, previousMotionGray);
      final (_, diffThresh) = cv.threshold(diff, 18, 255, cv.THRESH_BINARY);
      final diffOpen = cv.morphologyEx(diffThresh, cv.MORPH_OPEN, k7);
      final diffClose = cv.morphologyEx(diffOpen, cv.MORPH_CLOSE, k11);
      motionMask = cv.bitwiseOR(fgClose, diffClose);
      fgClose.dispose();
      diff.dispose();
      diffThresh.dispose();
      diffOpen.dispose();
      diffClose.dispose();
    }
    previousMotionGray?.dispose();
    previousMotionGray = motionGray;

    final cupTracks = tracks.values
        .where((t) => t.label != 'mover')
        .toList(growable: false);
    final fallenBurnZones = <cv.Rect>[
      for (final t in cupTracks)
        if (t.fallFrame != null &&
            (frameIdx - t.fallFrame!) <= _impactBurnFrames &&
            t.fallBox != null)
          t.fallBox!,
      for (final t in cupTracks)
        if (t.fallFrame == null && t.fallCounter > 0) t.box,
    ];

    final existingMovers = tracks.values
        .where((t) => t.label == 'mover')
        .toList();

    final detectorMoverBlobs = semanticDetections
        .where((d) => d.label == 'rc_car')
        .map(
          (d) => _Blob(
            d.box,
            (d.box.width * d.box.height * d.score).toDouble(),
            0.85,
            semantic: true,
          ),
        )
        .where((b) => b.box.width > 22 && b.box.height > 18)
        .toList();

    final motionMoverBlobs = _allBlobsRaw(motionMask, _minMoverArea)
        .where((b) => b.box.width > 22 && b.box.height > 18)
        .where((b) => !_targetSuppressesMover(b.box, cupTracks))
        .where((b) => !_isTargetOnlyMotion(b, cupTracks))
        .where((b) => !_overlapsBurnZone(b.box, fallenBurnZones))
        .where(
          (b) => _isPlausibleMotionCar(
            frame,
            b.box,
            cupTracks,
            hasActiveMover: existingMovers.isNotEmpty,
          ),
        )
        .take(_maxMoverTracks * 2)
        .toList(growable: false);

    final yoloCarLocked = detectorMoverBlobs.isNotEmpty;
    final allowPureMotionMoverSeed =
        !yoloCarLocked &&
        (semanticDetector == null || existingMovers.isNotEmpty);
    final filteredMoverBlobs = _rankMoverCandidates(
      _mergeMoverCandidates([
        ...detectorMoverBlobs,
        if (allowPureMotionMoverSeed) ...motionMoverBlobs,
      ]),
      existingMovers,
    ).take(_maxMoverTracks).toList();

    visibleCounts['mover'] = filteredMoverBlobs.length;

    _matchAndUpdate(
      tracks: tracks,
      existing: existingMovers,
      blobs: filteredMoverBlobs,
      label: 'mover',
      frameIdx: frameIdx,
      sourceFps: fps,
      effectiveFps: outputFps,
      frameStep: frameStep,
      maxJump: maxJumpPx,
      mintId: mintTrackId,
      emitLog: emitLog,
      canSpawn: existingMovers.length < _maxMoverTracks,
      isMover: true,
    );

    final moverIdsToRetire = tracks.entries
        .where(
          (e) =>
              e.value.label == 'mover' &&
              e.value.missingFrames > _disappearThresh * 6 &&
              e.value.path.isNotEmpty,
        )
        .map((e) => e.key)
        .toList();
    for (final id in moverIdsToRetire) {
      final t = tracks.remove(id)!;
      retiredMovers.add(
        _RetiredMover(
          id: t.id,
          lastBox: t.box,
          lastArea: t.area,
          lastCx: t.smCx > 0 ? t.smCx : t.cx,
          lastCy: t.smCy > 0 ? t.smCy : t.cy,
          lastVx: t.lastVx,
          lastVy: t.lastVy,
          retiredAtFrame: frameIdx,
          pathSegments: t.pathSegments
              .map((seg) => List<(int, int)>.from(seg))
              .toList(),
          bestConfidence: _pathConfidence(t),
        ),
      );
    }
    retiredMovers.removeWhere(
      (r) => (frameIdx - r.retiredAtFrame) > _moverRetireTtl,
    );

    final stale = tracks.entries
        .where(
          (e) =>
              (e.value.label != 'mover' &&
                  e.value.fallFrame == null &&
                  e.value.missingFrames > _disappearThresh * 3 &&
                  !e.value.baselineSet) ||
              (e.value.label == 'mover' &&
                  e.value.missingFrames > _disappearThresh * 6 &&
                  e.value.path.isEmpty),
        )
        .map((e) => e.key)
        .toList();
    for (final id in stale) {
      tracks.remove(id);
    }

    final newlySpawnedMovers = tracks.values
        .where(
          (t) =>
              t.label == 'mover' &&
              t.firstFrame == frameIdx &&
              t.path.length <= 1,
        )
        .toList();
    for (final spawned in newlySpawnedMovers) {
      _RetiredMover? best;
      double bestScore = double.infinity;
      for (final r in retiredMovers) {
        final dt = (frameIdx - r.retiredAtFrame).clamp(1, _moverRetireTtl);
        final predX = r.lastCx + r.lastVx * dt;
        final predY = r.lastCy + r.lastVy * dt;
        final dx = spawned.cx - predX;
        final dy = spawned.cy - predY;
        final dist = math.sqrt(dx * dx + dy * dy);
        final radius = math.min(
          0.5 * math.max(W, H).toDouble(),
          80 + dt.toDouble() * 4.0,
        );
        if (dist > radius) continue;
        final areaDelta = (spawned.area - r.lastArea).abs() / r.lastArea;
        if (areaDelta > _moverReIdAreaTol) continue;
        final score = dist + areaDelta * 200.0;
        if (score < bestScore) {
          bestScore = score;
          best = r;
        }
      }
      if (best != null) {
        final retired = best;
        retiredMovers.remove(retired);
        tracks.remove(spawned.id);
        final grafted = _Track(
          id: retired.id,
          label: 'mover',
          box: spawned.box,
          area: spawned.area,
          ratio: spawned.ratio,
          firstFrame: retired.retiredAtFrame,
        );
        grafted.confidence = math.max(
          spawned.confidence,
          retired.bestConfidence * 0.85,
        );
        grafted.smCx = spawned.cx;
        grafted.smCy = spawned.cy;
        grafted.trackedFrames = 1;
        grafted.lastSeenFrame = frameIdx;
        grafted.pathSegments
          ..clear()
          ..addAll(retired.pathSegments)
          ..add(<(int, int)>[]);
        tracks[grafted.id] = grafted;
        emitLog(
          frameIdx,
          frameIdx / fps,
          '${grafted.id} re-identified after re-entry',
          'success',
          color: 'rc_car',
        );
      }
    }

    final qualifiedColorTracks = tracks.values
        .where((t) => _isQualifiedTargetTrack(t, W, H, minDisplayConfidence))
        .toList();
    final fallenColorTracks =
        qualifiedColorTracks.where((t) => t.fallFrame != null).toList()
          ..sort((a, b) => a.fallFrame!.compareTo(b.fallFrame!));

    final allSegments = <List<(int, int)>>[];
    final sortedRetirees = [...retiredMovers]
      ..sort((a, b) => a.retiredAtFrame.compareTo(b.retiredAtFrame));
    for (final r in sortedRetirees) {
      allSegments.addAll(r.pathSegments);
    }
    final primaryMover = _primaryMover(tracks.values);
    if (primaryMover != null) {
      allSegments.addAll(primaryMover.pathSegments);
    }
    final motionPathPoints = allSegments.fold<int>(
      0,
      (sum, seg) => sum + seg.length,
    );
    final impactGuidedPath = semanticDetector == null
        ? _impactGuidedPath(fallenColorTracks)
        : const <(int, int)>[];
    final useImpactGuidedPath =
        motionPathPoints < 8 && impactGuidedPath.length >= _minMoverPathPoints;
    final renderSegments = useImpactGuidedPath
        ? <List<(int, int)>>[impactGuidedPath]
        : allSegments;
    if (renderSegments.isNotEmpty) {
      final segs = renderSegments;
      final shadowColor = cv.Scalar(0, 0, 0);
      final haloColor = cv.Scalar(255, 255, 255);
      final solidColor = useImpactGuidedPath
          ? cv.Scalar(255, 230, 30)
          : cv.Scalar(255, 210, 40);
      for (final seg in segs) {
        if (seg.length < 2) continue;
        final pts = seg.map((p) => cv.Point(p.$1, p.$2)).toList();
        cv.polylines(
          frame,
          cv.VecVecPoint.fromList([pts]),
          false,
          shadowColor,
          thickness: 8,
        );
        cv.polylines(
          frame,
          cv.VecVecPoint.fromList([pts]),
          false,
          haloColor,
          thickness: 5,
        );
        cv.polylines(
          frame,
          cv.VecVecPoint.fromList([pts]),
          false,
          solidColor,
          thickness: 3,
        );
        for (int i = 0; i < seg.length; i += 4) {
          final p = seg[i];
          cv.circle(frame, cv.Point(p.$1, p.$2), 5, shadowColor, thickness: -1);
          cv.circle(frame, cv.Point(p.$1, p.$2), 3, solidColor, thickness: -1);
        }
        if (useImpactGuidedPath && seg.length >= 2) {
          final p = seg.last;
          _drawLabel(frame, 'RC path', p.$1 + 8, p.$2 - 8, solidColor);
        }
      }
    }
    for (final t in tracks.values.where((t) => t.label == 'mover')) {
      if (debugMode) {
        cv.rectangle(frame, t.box, cv.Scalar(255, 80, 0), thickness: 3);
        _drawLabel(
          frame,
          'RC Car ${(t.confidence * 100).toStringAsFixed(0)}%',
          t.box.x,
          t.box.y,
          cv.Scalar(255, 80, 0),
        );
      }
    }

    final fallRank = {
      for (var e in fallenColorTracks.asMap().entries) e.value.id: e.key + 1,
    };

    for (final t in qualifiedColorTracks) {
      final visibleEnough = t.confidence >= minDisplayConfidence;
      final recentlyMatched = frameIdx - t.lastSeenFrame <= frameStep * 4;
      if ((!visibleEnough || !recentlyMatched) && t.fallFrame == null) {
        continue;
      }
      if (t.fallFrame != null && !recentlyMatched && !debugMode) {
        continue;
      }
      final fallen = t.fallFrame != null;
      final color = fallen
          ? cv.Scalar(0, 255, 80)
          : _bgrFromPacked(activeSpecs[t.label]!.displayBgrPacked);
      final rank = fallRank[t.id];
      final name = _targetName(t);
      final pct = (t.confidence * 100).toStringAsFixed(0);
      final label = (fallen && rank != null)
          ? '$name #$rank${debugMode ? " ($pct%)" : ""}'
          : '$name${debugMode ? " ($pct%)" : ""}';
      cv.rectangle(frame, t.box, color, thickness: 2);
      _drawLabel(frame, label, t.box.x, t.box.y, color);
    }

    for (final t in fallenColorTracks) {
      final fb = t.fallBox;
      if (fb == null) continue;
      final cx = fb.x + fb.width ~/ 2;
      final cy = fb.y + fb.height ~/ 2;
      const arm = 14;
      final halo = cv.Scalar(255, 255, 255);
      final stroke = cv.Scalar(0, 255, 80);
      cv.line(
        frame,
        cv.Point(cx - arm, cy - arm),
        cv.Point(cx + arm, cy + arm),
        halo,
        thickness: 5,
      );
      cv.line(
        frame,
        cv.Point(cx - arm, cy + arm),
        cv.Point(cx + arm, cy - arm),
        halo,
        thickness: 5,
      );
      cv.line(
        frame,
        cv.Point(cx - arm, cy - arm),
        cv.Point(cx + arm, cy + arm),
        stroke,
        thickness: 2,
      );
      cv.line(
        frame,
        cv.Point(cx - arm, cy + arm),
        cv.Point(cx + arm, cy - arm),
        stroke,
        thickness: 2,
      );
      cv.circle(frame, cv.Point(cx, cy), 3, stroke, thickness: -1);
    }

    cv.putText(
      frame,
      'Score: ${fallenColorTracks.length}/${qualifiedColorTracks.length}',
      cv.Point(10, H - 25),
      cv.FONT_HERSHEY_SIMPLEX,
      1.0,
      cv.Scalar(80, 255, 80),
      thickness: 3,
    );

    writer?.write(frame);

    var stopAfterFrame = false;
    if (earlyStopOnComplete) {
      final livePathPoints = primaryMover?.path.length ?? 0;
      final retiredPathPoints = retiredMovers.fold<int>(
        0,
        (sum, r) =>
            sum + r.pathSegments.fold<int>(0, (s, seg) => s + seg.length),
      );
      final stopTargetCount =
          expectedTargetCount ?? qualifiedColorTracks.length;
      final allTargetsScored =
          stopTargetCount > 0 && fallenColorTracks.length >= stopTargetCount;
      final pathSeen =
          (livePathPoints + retiredPathPoints) >= _minMoverPathPoints;
      if (allTargetsScored && pathSeen) {
        completeAtFrame ??= frameIdx;
        final graceFrames = math.max(frameStep, (fps * 0.75).round());
        stopAfterFrame = frameIdx - completeAtFrame >= graceFrames;
      }
    }

    processedFrameCount++;

    if (frameIdx % _sampleEveryFrames == 0) {
      emitCount(frameIdx / fps, Map<String, int>.from(visibleCounts));
    }

    final nowAfterFrame = DateTime.now();
    if (nowAfterFrame.difference(lastAnnotateProgressAt).inMilliseconds >=
            900 ||
        processedFrameCount == totalProcessFrames) {
      final p = totalProcessFrames > 0
          ? (processedFrameCount / totalProcessFrames).clamp(0.01, 0.99)
          : 0.01;
      emitProgress(
        PipelineStage.annotate,
        p,
        totalProcessFrames > 0
            ? 'Processed $processedFrameCount / $totalProcessFrames frames'
            : 'Processed $processedFrameCount frames',
      );
      lastAnnotateProgressAt = nowAfterFrame;
    }

    if (!identical(targetFrame, frame)) targetFrame.dispose();
    if (!identical(carFrame, frame)) carFrame.dispose();
    frame.dispose();
    if (stopAfterFrame) {
      emitLog(
        frameIdx,
        frameIdx / fps,
        'Fast stop: all expected targets scored and path captured',
        'success',
      );
      break;
    }
    frameIdx++;
  }

  cap.release();
  writer?.release();
  semanticDetector?.close();
  previousMotionGray?.dispose();
  k7.dispose();
  k11.dispose();

  emitProgress(PipelineStage.complete, 1.0, 'Done');

  final colorTracks = tracks.values
      .where((t) => _isQualifiedTargetTrack(t, W, H, minDisplayConfidence))
      .toList();
  final fallenSorted = colorTracks.where((t) => t.fallFrame != null).toList()
    ..sort((a, b) => a.fallFrame!.compareTo(b.fallFrame!));

  final activeIds = colorTracks.map(_targetKey).toList();
  final fallingOrder = fallenSorted.map(_targetKey).toList();

  final confidences = <String, double>{
    for (final t in colorTracks) _targetKey(t): t.confidence.clamp(0.0, 1.0),
  };
  final primaryMover = _primaryMover(tracks.values);

  int totalPathPoints = primaryMover?.path.length ?? 0;
  for (final r in retiredMovers) {
    totalPathPoints += r.pathSegments.fold<int>(0, (s, seg) => s + seg.length);
  }
  final impactGuidedPath = semanticDetector == null
      ? _impactGuidedPath(fallenSorted)
      : const <(int, int)>[];
  if (totalPathPoints < 8 && impactGuidedPath.length >= _minMoverPathPoints) {
    totalPathPoints = impactGuidedPath.length;
  }

  double bestConf = 0.0;
  if (primaryMover != null && primaryMover.path.length >= _minMoverPathPoints) {
    bestConf = _pathConfidence(primaryMover);
  }
  for (final r in retiredMovers) {
    if (r.bestConfidence > bestConf) bestConf = r.bestConfidence;
  }
  if (impactGuidedPath.length >= _minMoverPathPoints) {
    bestConf = math.max(bestConf, 0.82);
  }
  if (totalPathPoints >= _minMoverPathPoints) {
    confidences['rc_car'] = bestConf.clamp(0.0, 1.0);
  }

  final targetColors = <String, int>{};
  for (final t in colorTracks) {
    targetColors[_targetKey(t)] =
        activeSpecs[t.label]?.displayBgrPacked ?? 0xFF888888;
  }
  if (totalPathPoints >= _minMoverPathPoints) {
    targetColors['rc_car'] = 0xFF22D3EE;
  }

  final invScale = scale > 0 ? 1.0 / scale : 1.0;
  final fallEventsMap = <String, Map<String, dynamic>>{};
  for (final t in fallenSorted) {
    final box = t.fallBox ?? t.box;
    final cx = (box.x + box.width / 2.0) * invScale;
    final cy = (box.y + box.height / 2.0) * invScale;
    final ox = (box.x * invScale).round().clamp(0, origW - 1);
    final oy = (box.y * invScale).round().clamp(0, origH - 1);
    final ow = (box.width * invScale).round().clamp(1, origW);
    final oh = (box.height * invScale).round().clamp(1, origH);
    fallEventsMap[_targetKey(t)] = {
      'x': cx.round().clamp(0, origW - 1),
      'y': cy.round().clamp(0, origH - 1),
      'box_x': ox,
      'box_y': oy,
      'box_w': ow,
      'box_h': oh,
      'frame': t.fallFrame!,
      't': t.fallFrame! / fps,
      'signal': t.fallSignal ?? 'unknown',
    };
  }

  send.send({
    'type': 'done',
    'result': {
      'score': fallenSorted.length,
      'total_targets': colorTracks.where((t) => t.baselineSet).length,
      'active_targets': activeIds,
      'falling_order': fallingOrder,
      'fall_times_sec': {
        for (final t in fallenSorted) _targetKey(t): t.fallFrame! / fps,
      },
      'fall_events': fallEventsMap,
      'confidences': confidences,
      'count_timeline': countTimeline,
      'event_log': eventLog,
      'target_colors': targetColors,
      'frame_width': W,
      'frame_height': H,
      'original_width': origW,
      'original_height': origH,
      'car_path_points': totalPathPoints,
      'path_drawn': totalPathPoints >= _minMoverPathPoints,
      'frames_processed': frameIdx,
      'fps': fps,
      'output_video': writerHandle.isOpen ? writerHandle.outputPath : '',
      'processing_time_ms': DateTime.now().difference(t0).inMilliseconds,
      'preprocessing': preprocessProfile.toJson(),
    },
  });
}

_WriterHandle _openBestWriter(
  String requestedPath,
  double fps,
  int width,
  int height,
) {
  final base = _withoutExtension(requestedPath);
  final candidates = <(String, String)>[
    (requestedPath, 'mp4v'),
    ('$base.avi', 'MJPG'),
    ('$base-xvid.avi', 'XVID'),
    ('$base-mjpg.mkv', 'MJPG'),
  ];

  for (final (path, codec) in candidates) {
    cv.VideoWriter? writer;
    try {
      writer = cv.VideoWriter.empty();
      writer.open(path, codec, fps, (width, height));
      if (writer.isOpened) {
        return _WriterHandle(writer: writer, outputPath: path, codec: codec);
      }
      writer.release();
    } catch (_) {
      try {
        writer?.release();
      } catch (_) {}
    }
  }

  return const _WriterHandle(writer: null, outputPath: '', codec: '');
}

String _withoutExtension(String path) {
  final slash = math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  final dot = path.lastIndexOf('.');
  if (dot > slash) return path.substring(0, dot);
  return path;
}

int _evenDimension(int value) {
  final safe = math.max(2, value);
  return safe.isEven ? safe : safe - 1;
}

String _targetKey(_Track track) => 'target_${track.targetNumber ?? track.id}';

String _targetName(_Track track) => 'Target ${track.targetNumber ?? track.id}';

void _matchAndUpdate({
  required Map<String, _Track> tracks,
  required List<_Track> existing,
  required List<_Blob> blobs,
  required String label,
  required int frameIdx,
  required double sourceFps,
  required double effectiveFps,
  required int frameStep,
  required double maxJump,
  required String Function(String) mintId,
  required void Function(int, double, String, String, {String? color}) emitLog,
  required bool canSpawn,
  bool isMover = false,
}) {
  final pairs = <(int, double, int)>[];
  for (int t = 0; t < existing.length; t++) {
    final tr = existing[t];
    for (int b = 0; b < blobs.length; b++) {
      final bl = blobs[b];
      if (!isMover && !tr.baselineSet && !_isBaselineCompatibleTarget(tr, bl)) {
        continue;
      }
      final cx = bl.box.x + bl.box.width / 2.0;
      final cy = bl.box.y + bl.box.height / 2.0;
      final anchorCx = isMover && tr.smCx > 0 ? tr.smCx + tr.lastVx : tr.cx;
      final anchorCy = isMover && tr.smCy > 0 ? tr.smCy + tr.lastVy : tr.cy;
      final d = math.sqrt(
        (cx - anchorCx) * (cx - anchorCx) + (cy - anchorCy) * (cy - anchorCy),
      );
      final jumpGate = isMover && bl.semantic
          ? maxJump * _semanticMoverJumpMultiplier
          : maxJump;
      if (d > jumpGate) continue;
      if (isMover && tr.area > 1 && !bl.semantic) {
        final ratio = bl.area / tr.area;
        if (ratio > _maxMoverAreaJump || ratio < 1.0 / _maxMoverAreaJump) {
          continue;
        }
      }
      pairs.add((t, d, b));
    }
  }
  pairs.sort((a, b) => a.$2.compareTo(b.$2));

  final usedT = <int>{}, usedB = <int>{};
  for (final (t, _, b) in pairs) {
    if (usedT.contains(t) || usedB.contains(b)) continue;
    usedT.add(t);
    usedB.add(b);
    _updateTrack(
      existing[t],
      blobs[b],
      frameIdx,
      sourceFps,
      effectiveFps,
      frameStep,
      isMover,
      emitLog,
    );
  }

  for (int t = 0; t < existing.length; t++) {
    if (usedT.contains(t)) continue;
    final tr = existing[t];
    tr.missingFrames++;
    tr.confidence *= 0.85;
    if (tr.baselineSet &&
        !isMover &&
        tr.fallFrame == null &&
        tr.trackedFrames / effectiveFps >= _minVanishFallAgeSec &&
        tr.missingFrames == _disappearThresh) {
      emitLog(
        frameIdx,
        frameIdx / sourceFps,
        '${tr.id} temporarily hidden — not counted as fallen',
        'warn',
        color: _targetKey(tr),
      );
    }
  }

  if (canSpawn) {
    for (int b = 0; b < blobs.length; b++) {
      if (usedB.contains(b)) continue;
      final bl = blobs[b];
      final id = mintId(label);
      final tr = _Track(
        id: id,
        label: label,
        box: bl.box,
        area: bl.area,
        ratio: bl.box.width > 0 ? bl.box.height / bl.box.width.toDouble() : 0,
        firstFrame: frameIdx,
      );
      tr.confidence = _confidence(bl, bl.area);
      tr.lastSeenFrame = frameIdx;
      if (isMover) {
        tr.trackedFrames = 1;
        tr.smCx = tr.cx;
        tr.smCy = tr.cy;
        _appendPathPoint(
          tr.pathSegments.last,
          tr.smCx.round(),
          tr.smCy.round(),
        );
      }
      tracks[id] = tr;
    }
  }
}

void _updateTrack(
  _Track tr,
  _Blob bl,
  int frameIdx,
  double sourceFps,
  double effectiveFps,
  int frameStep,
  bool isMover,
  void Function(int, double, String, String, {String? color}) emitLog,
) {
  tr.box = bl.box;
  tr.area = bl.area;
  tr.ratio = bl.box.width > 0 ? bl.box.height / bl.box.width.toDouble() : 0;
  if (bl.area > tr.maxArea) tr.maxArea = bl.area;
  tr.confidence = _confidence(bl, tr.maxArea);
  tr.missingFrames = 0;
  tr.trackedFrames++;
  tr.lastSeenFrame = frameIdx;

  if (isMover) {
    final cx = tr.cx, cy = tr.cy;
    final prevSmCx = tr.smCx;
    final prevSmCy = tr.smCy;
    final distFromPath = prevSmCx >= 0
        ? math.sqrt(
            (cx - prevSmCx) * (cx - prevSmCx) +
                (cy - prevSmCy) * (cy - prevSmCy),
          )
        : 0.0;
    final snapGate = math.max(
      56.0,
      math.max(tr.box.width, tr.box.height).toDouble() * 1.20,
    );
    if (bl.semantic && prevSmCx >= 0 && distFromPath > snapGate) {
      final currentSeg = tr.pathSegments.last;
      if (currentSeg.isNotEmpty) tr.pathSegments.add(<(int, int)>[]);
      tr.smCx = cx;
      tr.smCy = cy;
      tr.lastVx = 0;
      tr.lastVy = 0;
      _appendPathPoint(tr.pathSegments.last, tr.smCx.round(), tr.smCy.round());
      emitLog(
        frameIdx,
        frameIdx / sourceFps,
        '${tr.id} snapped back to YOLO RC-car detection',
        'success',
        color: 'rc_car',
      );
      return;
    }
    if (tr.smCx < 0) {
      tr.smCx = cx;
      tr.smCy = cy;
    } else {
      tr.smCx = tr.smCx * 0.55 + cx * 0.45;
      tr.smCy = tr.smCy * 0.55 + cy * 0.45;
    }
    tr.lastVx = prevSmCx >= 0 ? tr.smCx - prevSmCx : 0;
    tr.lastVy = prevSmCy >= 0 ? tr.smCy - prevSmCy : 0;
    final currentSeg = tr.pathSegments.last;
    _appendPathPoint(currentSeg, tr.smCx.round(), tr.smCy.round());
    return;
  }

  if (!tr.baselineSet && tr.trackedFrames >= _minTrackedFrames) {
    tr.initialRatio = tr.ratio;
    tr.initialArea = tr.area;
    tr.initialCx = tr.cx;
    tr.initialCy = tr.cy;
    tr.initialWidth = tr.box.width.toDouble();
    tr.initialHeight = tr.box.height.toDouble();
    tr.baselineBox = tr.box;
    tr.baselineSet = true;
    tr.fallCounter = 0;
    emitLog(
      frameIdx,
      frameIdx / sourceFps,
      '${tr.id} baseline locked '
          '(ratio=${tr.ratio.toStringAsFixed(2)}, area=${tr.area.toStringAsFixed(0)})',
      'info',
      color: _targetKey(tr),
    );
    return;
  }

  if (tr.baselineSet && tr.fallFrame == null) {
    final ageSec = tr.trackedFrames / effectiveFps;
    final ratioFallen =
        ageSec >= _minShapeFallAgeSec &&
        tr.ratio < tr.initialRatio * _fallRatioFactor &&
        tr.initialRatio > 0.5;
    final areaFallen =
        ageSec >= _minAreaFallAgeSec &&
        tr.area < tr.initialArea * _fallAreaFactor;
    final displacement = math.sqrt(
      (tr.cx - tr.initialCx) * (tr.cx - tr.initialCx) +
          (tr.cy - tr.initialCy) * (tr.cy - tr.initialCy),
    );
    final spreadFallen =
        ageSec >= _minShapeFallAgeSec &&
        (tr.box.width > tr.initialWidth * 1.45 ||
            tr.box.height > tr.initialHeight * 1.45);
    final motionShapeChanged =
        tr.area < tr.initialArea * 0.68 ||
        tr.ratio < tr.initialRatio * 0.72 ||
        tr.box.width > tr.initialWidth * 1.30 ||
        tr.box.height > tr.initialHeight * 1.30;
    final displacementFallen =
        ageSec >= _minShapeFallAgeSec &&
        motionShapeChanged &&
        displacement >
            math.max(32.0, math.max(tr.initialWidth, tr.initialHeight) * 0.45);
    if (ratioFallen || areaFallen || displacementFallen || spreadFallen) {
      tr.fallCounter++;
      if (tr.fallCounter >= _fallConfirmFrames) {
        tr.fallFrame = frameIdx - (_fallConfirmFrames - 1) * frameStep;
        tr.fallBox = tr.box;
        tr.fallSignal = ratioFallen
            ? 'shape'
            : areaFallen
            ? 'area'
            : displacementFallen
            ? 'motion'
            : 'spread';
        emitLog(
          frameIdx,
          tr.fallFrame! / sourceFps,
          '${tr.id} FELL (${tr.fallSignal} signal)',
          'event',
          color: _targetKey(tr),
        );
      }
    } else {
      if (tr.fallCounter > 0) tr.fallCounter--;
      tr.initialArea = tr.initialArea * 0.97 + tr.area * 0.03;
    }
  }
}

void _appendPathPoint(List<(int, int)> path, int x, int y) {
  if (path.isEmpty) {
    path.add((x, y));
    return;
  }
  final last = path.last;
  final dx = x - last.$1;
  final dy = y - last.$2;
  if (dx * dx + dy * dy >= 25) {
    path.add((x, y));
  }
}

List<(int, int)> _impactGuidedPath(List<_Track> fallenTracks) {
  if (fallenTracks.length < 2) return const <(int, int)>[];

  final controls = <(int, int)>[];
  for (final t in fallenTracks) {
    final box = t.fallBox ?? t.box;
    final point = (box.x + box.width ~/ 2, box.y + box.height ~/ 2);
    if (controls.isEmpty) {
      controls.add(point);
      continue;
    }
    final last = controls.last;
    final dx = point.$1 - last.$1;
    final dy = point.$2 - last.$2;
    if (dx * dx + dy * dy > 100) controls.add(point);
  }
  if (controls.length < 2) return const <(int, int)>[];

  if (controls.length == 2) {
    return _densifyLine(controls.first, controls.last, samples: 18);
  }

  final smoothed = <(int, int)>[];
  for (int i = 0; i < controls.length - 1; i++) {
    final p0 = controls[(i - 1).clamp(0, controls.length - 1)];
    final p1 = controls[i];
    final p2 = controls[i + 1];
    final p3 = controls[(i + 2).clamp(0, controls.length - 1)];
    for (int s = 0; s < 10; s++) {
      final t = s / 10.0;
      final tt = t * t;
      final ttt = tt * t;
      final x =
          0.5 *
          ((2 * p1.$1) +
              (-p0.$1 + p2.$1) * t +
              (2 * p0.$1 - 5 * p1.$1 + 4 * p2.$1 - p3.$1) * tt +
              (-p0.$1 + 3 * p1.$1 - 3 * p2.$1 + p3.$1) * ttt);
      final y =
          0.5 *
          ((2 * p1.$2) +
              (-p0.$2 + p2.$2) * t +
              (2 * p0.$2 - 5 * p1.$2 + 4 * p2.$2 - p3.$2) * tt +
              (-p0.$2 + 3 * p1.$2 - 3 * p2.$2 + p3.$2) * ttt);
      _appendPathPoint(smoothed, x.round(), y.round());
    }
  }
  _appendPathPoint(smoothed, controls.last.$1, controls.last.$2);
  return smoothed;
}

List<(int, int)> _densifyLine((int, int) a, (int, int) b, {int samples = 12}) {
  final out = <(int, int)>[];
  for (int i = 0; i <= samples; i++) {
    final t = i / samples;
    final x = a.$1 + (b.$1 - a.$1) * t;
    final y = a.$2 + (b.$2 - a.$2) * t;
    _appendPathPoint(out, x.round(), y.round());
  }
  return out;
}

int _ov1D(int a1, int a2, int b1, int b2) =>
    (a2.clamp(b1, b2) - a1.clamp(b1, b2)).clamp(0, 999999);

List<_Blob> _mergeMoverCandidates(Iterable<_Blob> candidates) {
  final sorted = candidates.toList()
    ..sort((a, b) {
      if (a.semantic != b.semantic) return a.semantic ? -1 : 1;
      return b.area.compareTo(a.area);
    });
  final out = <_Blob>[];
  for (final candidate in sorted) {
    final duplicate = out.any(
      (kept) =>
          _rectOverlapFrac(candidate.box, kept.box) > 0.35 ||
          _centerDistance(candidate.box, kept.box) <
              math.max(18.0, (candidate.box.width + kept.box.width) * 0.35),
    );
    if (!duplicate) out.add(candidate);
  }
  return out;
}

List<_Blob> _rankMoverCandidates(
  List<_Blob> candidates,
  List<_Track> existing,
) {
  final current =
      existing.where((t) => t.path.isNotEmpty || t.trackedFrames > 0).toList()
        ..sort((a, b) => b.confidence.compareTo(a.confidence));
  final anchor = current.isNotEmpty ? current.first : null;
  candidates.sort((a, b) {
    final sa = _moverCandidateScore(a, anchor);
    final sb = _moverCandidateScore(b, anchor);
    return sa.compareTo(sb);
  });
  return candidates;
}

double _moverCandidateScore(_Blob blob, _Track? anchor) {
  final semanticBonus = blob.semantic ? -120.0 : 0.0;
  if (anchor == null) {
    return semanticBonus - math.sqrt(blob.area.clamp(1.0, double.infinity));
  }

  final cx = blob.box.x + blob.box.width / 2.0;
  final cy = blob.box.y + blob.box.height / 2.0;
  final predX = (anchor.smCx > 0 ? anchor.smCx : anchor.cx) + anchor.lastVx;
  final predY = (anchor.smCy > 0 ? anchor.smCy : anchor.cy) + anchor.lastVy;
  final dx = cx - predX;
  final dy = cy - predY;
  final dist = math.sqrt(dx * dx + dy * dy);
  final areaDelta = anchor.area > 1
      ? (blob.area - anchor.area).abs() / anchor.area
      : 0.0;
  return semanticBonus + dist + areaDelta * 85.0;
}

bool _isPlausibleSemanticCar(cv.Mat frame, cv.Rect box) {
  final x0 = box.x.clamp(0, math.max(0, frame.cols - 1)).toInt();
  final y0 = box.y.clamp(0, math.max(0, frame.rows - 1)).toInt();
  final x1 = (box.x + box.width).clamp(x0 + 1, frame.cols).toInt();
  final y1 = (box.y + box.height).clamp(y0 + 1, frame.rows).toInt();
  final w = x1 - x0;
  final h = y1 - y0;
  if (w < 22 || h < 18) return false;
  if (h / w > 1.45) return false;

  final channels = frame.rows > 0 && frame.cols > 0
      ? frame.data.length ~/ (frame.rows * frame.cols)
      : 0;
  if (channels < 3) return true;

  final stepX = math.max(1, w ~/ 14);
  final stepY = math.max(1, h ~/ 14);
  var samples = 0;
  var saturated = 0;
  var dark = 0;
  var neutralBright = 0;
  final data = frame.data;
  for (int y = y0; y < y1; y += stepY) {
    for (int x = x0; x < x1; x += stepX) {
      final idx = (y * frame.cols + x) * channels;
      if (idx + 2 >= data.length) continue;
      final b = data[idx];
      final g = data[idx + 1];
      final r = data[idx + 2];
      final maxV = math.max(r, math.max(g, b));
      final minV = math.min(r, math.min(g, b));
      final chroma = maxV - minV;
      samples++;
      if (maxV < 95) dark++;
      if (maxV > 120 && chroma < 60) neutralBright++;
      if (chroma > 70 && maxV > 75) saturated++;
    }
  }
  if (samples == 0) return true;

  final saturatedRatio = saturated / samples;
  final darkRatio = dark / samples;
  final neutralRatio = neutralBright / samples;
  if (saturatedRatio > 0.62 && darkRatio < 0.06 && neutralRatio < 0.08) {
    return false;
  }
  return true;
}

bool _isBaselineCompatibleTarget(_Track track, _Blob blob) {
  final ratio = blob.box.height / math.max(1, blob.box.width);
  if (ratio < 0.70 || ratio > 5.4) return false;
  if (blob.fillRatio < 0.18) return false;

  final areaRatio = blob.area / track.area.clamp(1.0, double.infinity);
  if (areaRatio < 0.35 || areaRatio > 2.75) return false;

  final dist = _centerDistance(track.box, blob.box);
  final gate = math.max(30.0, (track.box.width + blob.box.width) * 0.70);
  return dist <= gate || _rectOverlapFrac(track.box, blob.box) > 0.22;
}

bool _isPlausibleMotionCar(
  cv.Mat frame,
  cv.Rect box,
  Iterable<_Track> targetTracks, {
  required bool hasActiveMover,
}) {
  if (!_isPlausibleSemanticCar(frame, box)) return false;
  final w = box.width.toDouble();
  final h = box.height.toDouble();
  if (w < 24 || h < 18) return false;
  final aspect = w / h;
  if (aspect < 0.65 || aspect > 5.2) return false;

  final boxArea = w * h;
  double largestTargetArea = 0;
  for (final t in targetTracks) {
    if (t.baselineSet) {
      largestTargetArea = math.max(
        largestTargetArea,
        (t.box.width * t.box.height).toDouble(),
      );
    }
  }

  if (!hasActiveMover) {
    if (boxArea < 900) return false;
    if (largestTargetArea > 0 && boxArea < largestTargetArea * 0.55) {
      return false;
    }
  }
  return true;
}

bool _targetSuppressesMover(cv.Rect box, Iterable<_Track> targetTracks) {
  final blobArea = (box.width * box.height).toDouble();
  if (blobArea <= 0) return false;
  final cx = box.x + box.width / 2.0;
  final cy = box.y + box.height / 2.0;

  for (final t in targetTracks) {
    final impactLike = t.fallFrame != null || t.fallCounter > 0;
    final zones = <cv.Rect>[
      t.box,
      if (t.baselineBox != null) t.baselineBox!,
      if (t.fallBox != null) t.fallBox!,
    ];
    for (final z in zones) {
      final targetArea = (z.width * z.height).toDouble().clamp(
        1.0,
        double.infinity,
      );
      final pad = impactLike ? _impactBurnPad : 10;
      final left = z.x - pad;
      final top = z.y - pad;
      final right = z.x + z.width + pad;
      final bottom = z.y + z.height + pad;
      final centerInZone =
          cx >= left && cx <= right && cy >= top && cy <= bottom;

      final iw = _ov1D(box.x, box.x + box.width, z.x, z.x + z.width).toDouble();
      final ih = _ov1D(
        box.y,
        box.y + box.height,
        z.y,
        z.y + z.height,
      ).toDouble();
      final overlap = iw * ih;

      if (impactLike && centerInZone && blobArea < targetArea * 3.2) {
        return true;
      }
      if (impactLike &&
          overlap > blobArea * 0.50 &&
          blobArea < targetArea * 3.5) {
        return true;
      }
      if (!impactLike &&
          overlap > blobArea * 0.82 &&
          blobArea < targetArea * 1.6) {
        return true;
      }
    }
  }
  return false;
}

bool _isTargetOnlyMotion(_Blob blob, Iterable<_Track> targetTracks) {
  final blobArea = (blob.box.width * blob.box.height).toDouble();
  if (blobArea <= 0) return false;

  for (final t in targetTracks) {
    final iw = _ov1D(
      blob.box.x,
      blob.box.x + blob.box.width,
      t.box.x,
      t.box.x + t.box.width,
    ).toDouble();
    final ih = _ov1D(
      blob.box.y,
      blob.box.y + blob.box.height,
      t.box.y,
      t.box.y + t.box.height,
    ).toDouble();
    final overlap = iw * ih;
    final targetArea = (t.box.width * t.box.height).toDouble();
    if (overlap > blobArea * 0.90 && blobArea < targetArea * 1.25) {
      return true;
    }
  }
  return false;
}

_Track? _primaryMover(Iterable<_Track> tracks) {
  final movers = tracks
      .where((t) => t.label == 'mover' && t.path.isNotEmpty)
      .toList();
  if (movers.isEmpty) return null;
  movers.sort((a, b) {
    final pathCmp = b.path.length.compareTo(a.path.length);
    if (pathCmp != 0) return pathCmp;
    return b.confidence.compareTo(a.confidence);
  });
  return movers.first;
}

double _pathConfidence(_Track mover) {
  final trajectoryScore = (mover.path.length / 80.0).clamp(0.0, 1.0);
  return math
      .max(mover.confidence, 0.55 + trajectoryScore * 0.43)
      .clamp(0.0, 0.98);
}

double _confidence(_Blob b, double maxAreaSeen) {
  final sizeScore = (b.area / maxAreaSeen).clamp(0.0, 1.0);
  final fillScore = ((b.fillRatio - _minFillRatio) / (1.0 - _minFillRatio))
      .clamp(0.0, 1.0);
  return (0.5 + (0.6 * sizeScore + 0.4 * fillScore) * 0.5).clamp(0.0, 1.0);
}

enum _DetectorFormat { ssdPostProcess, yoloRaw }

class _SemanticDetector {
  final Interpreter _interpreter;
  final List<String> _labels;
  final int _inputW;
  final int _inputH;
  final int _maxDetections;
  final _DetectorFormat _format;
  final TensorType _inputType;
  final List<int> _outputShape;
  Float32List? _floatInputBuffer;
  Float32List? _outputBuffer;

  _SemanticDetector._(
    this._interpreter,
    this._labels,
    this._inputW,
    this._inputH,
    this._maxDetections,
    this._format,
    this._inputType,
    this._outputShape,
  );

  static _SemanticDetector? tryCreate(
    Uint8List? modelBytes,
    List<String> labels,
  ) {
    if (modelBytes == null || modelBytes.isEmpty || labels.isEmpty) return null;
    if (!_hasCustomDetectorLabels(labels)) return null;
    try {
      final threads = Platform.isAndroid
          ? math.min(8, math.max(1, Platform.numberOfProcessors)).toInt()
          : 4;
      final options = InterpreterOptions()..threads = threads;
      if (Platform.isAndroid) {
        options.useNnApiForAndroid = true;
      }
      final interpreter = Interpreter.fromBuffer(modelBytes, options: options);
      final inputTensor = interpreter.getInputTensors().first;
      final inputShape = inputTensor.shape;
      if (inputShape.length != 4) {
        interpreter.close();
        return null;
      }
      final outputs = interpreter.getOutputTensors();
      final outputShape = outputs.first.shape;
      final format = outputs.length >= 4
          ? _DetectorFormat.ssdPostProcess
          : _DetectorFormat.yoloRaw;
      final maxDetections = format == _DetectorFormat.ssdPostProcess
          ? (outputShape.length >= 2 ? outputShape[1] : 10)
          : _yoloPredictionCount(outputShape);
      return _SemanticDetector._(
        interpreter,
        labels.map((l) => l.toLowerCase()).toList(),
        inputShape[2],
        inputShape[1],
        maxDetections,
        format,
        inputTensor.type,
        outputShape,
      );
    } catch (_) {
      return null;
    }
  }

  List<_SemanticDetection> detect(cv.Mat frame, {required double threshold}) {
    try {
      final resized = cv.resize(frame, (
        _inputW,
        _inputH,
      ), interpolation: cv.INTER_AREA);
      final rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB);
      resized.dispose();

      final input = _inputType == TensorType.float32
          ? _floatInputBytes(rgb.data)
          : rgb.data.reshape<int>([1, _inputH, _inputW, 3]);
      rgb.dispose();

      if (_format == _DetectorFormat.yoloRaw) {
        return _detectYoloRaw(input, frame, threshold);
      }
      return _detectSsd(input, frame, threshold);
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

  List<_SemanticDetection> _detectSsd(
    Object input,
    cv.Mat frame,
    double threshold,
  ) {
    try {
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

      final n = count.first.round().clamp(0, _maxDetections);
      final detections = <_SemanticDetection>[];
      for (int i = 0; i < n; i++) {
        final score = scores[0][i];
        if (score < threshold) continue;
        final label = _labelForClass(classes[0][i]);
        if (!_customDetectorLabels.contains(label)) continue;

        final b = boxes[0][i];
        final yMin = (b[0].clamp(0.0, 1.0) * frame.rows).round();
        final xMin = (b[1].clamp(0.0, 1.0) * frame.cols).round();
        final yMax = (b[2].clamp(0.0, 1.0) * frame.rows).round();
        final xMax = (b[3].clamp(0.0, 1.0) * frame.cols).round();
        final w = math.max(1, xMax - xMin);
        final h = math.max(1, yMax - yMin);
        detections.add(
          _SemanticDetection(label, score, cv.Rect(xMin, yMin, w, h)),
        );
      }
      return detections;
    } catch (_) {
      return const [];
    }
  }

  List<_SemanticDetection> _detectYoloRaw(
    Object input,
    cv.Mat frame,
    double threshold,
  ) {
    try {
      final output = _rawOutputBuffer();
      _interpreter.run(input, output.buffer);
      final decoded = _decodeYoloOutputFlat(output, frame, threshold);
      return _nmsDetections(decoded, 0.45);
    } catch (_) {
      return const [];
    }
  }

  List<_SemanticDetection> _decodeYoloOutputFlat(
    Float32List output,
    cv.Mat frame,
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

    final detections = <_SemanticDetection>[];
    for (int p = 0; p < predictions; p++) {
      if (!channelsFirst && channels == 6) {
        final score = valueAt(p, 4);
        if (score < threshold) continue;
        final label = _labelForClass(valueAt(p, 5));
        if (!_customDetectorLabels.contains(label)) continue;
        final box = _semanticRectFromCompactOutput(
          valueAt(p, 0),
          valueAt(p, 1),
          valueAt(p, 2),
          valueAt(p, 3),
          _inputW,
          _inputH,
          frame.cols,
          frame.rows,
        );
        if (box == null) continue;
        detections.add(_SemanticDetection(label, score, box));
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
      if (!_customDetectorLabels.contains(label)) continue;

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

      final xMin = ((cx - w / 2).clamp(0.0, 1.0) * frame.cols).round();
      final yMin = ((cy - h / 2).clamp(0.0, 1.0) * frame.rows).round();
      final xMax = ((cx + w / 2).clamp(0.0, 1.0) * frame.cols).round();
      final yMax = ((cy + h / 2).clamp(0.0, 1.0) * frame.rows).round();
      if (xMax <= xMin || yMax <= yMin) continue;
      detections.add(
        _SemanticDetection(
          label,
          bestScore,
          cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin),
        ),
      );
    }
    return detections;
  }

  List<_SemanticDetection> _decodeYoloOutput(
    Object output,
    cv.Mat frame,
    double threshold,
  ) {
    if (_outputShape.length != 3 || output is! List || output.isEmpty) {
      return const [];
    }
    final batch = output[0];
    if (batch is! List) return const [];

    final labelCount = _labels.length;
    final channelsFirst =
        _outputShape[1] == labelCount + 4 ||
        (_outputShape[1] < _outputShape[2] &&
            _outputShape[1] <= labelCount + 8);
    final channels = channelsFirst ? _outputShape[1] : _outputShape[2];
    final predictions = channelsFirst ? _outputShape[2] : _outputShape[1];
    if (channels < 5 || predictions <= 0) return const [];

    double valueAt(int pred, int ch) {
      if (channelsFirst) {
        return ((batch[ch] as List)[pred] as num).toDouble();
      }
      return ((batch[pred] as List)[ch] as num).toDouble();
    }

    final detections = <_SemanticDetection>[];
    for (int p = 0; p < predictions; p++) {
      if (!channelsFirst && channels == 6) {
        final score = valueAt(p, 4);
        if (score < threshold) continue;
        final label = _labelForClass(valueAt(p, 5));
        if (!_customDetectorLabels.contains(label)) continue;
        final box = _semanticRectFromCompactOutput(
          valueAt(p, 0),
          valueAt(p, 1),
          valueAt(p, 2),
          valueAt(p, 3),
          _inputW,
          _inputH,
          frame.cols,
          frame.rows,
        );
        if (box == null) continue;
        detections.add(_SemanticDetection(label, score, box));
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
      if (!_customDetectorLabels.contains(label)) continue;

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

      final xMin = ((cx - w / 2).clamp(0.0, 1.0) * frame.cols).round();
      final yMin = ((cy - h / 2).clamp(0.0, 1.0) * frame.rows).round();
      final xMax = ((cx + w / 2).clamp(0.0, 1.0) * frame.cols).round();
      final yMax = ((cy + h / 2).clamp(0.0, 1.0) * frame.rows).round();
      if (xMax <= xMin || yMax <= yMin) continue;
      detections.add(
        _SemanticDetection(
          label,
          bestScore,
          cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin),
        ),
      );
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

cv.Rect? _semanticRectFromCompactOutput(
  double x1,
  double y1,
  double x2,
  double y2,
  int inputW,
  int inputH,
  int frameW,
  int frameH,
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
  final left = (x1.clamp(0.0, 1.0) * frameW).round();
  final top = (y1.clamp(0.0, 1.0) * frameH).round();
  final right = (x2.clamp(0.0, 1.0) * frameW).round();
  final bottom = (y2.clamp(0.0, 1.0) * frameH).round();
  if (right <= left || bottom <= top) return null;
  return cv.Rect(left, top, right - left, bottom - top);
}

int _shapeProduct(List<int> shape) =>
    shape.fold<int>(1, (acc, v) => acc * math.max(1, v));

int _yoloPredictionCount(List<int> shape) {
  if (shape.length != 3) return 0;
  return math.max(shape[1], shape[2]);
}

List<_SemanticDetection> _nmsDetections(
  List<_SemanticDetection> detections,
  double iouThreshold,
) {
  final sorted = detections.toList()
    ..sort((a, b) => b.score.compareTo(a.score));
  final kept = <_SemanticDetection>[];
  for (final det in sorted) {
    final overlaps = kept.any(
      (k) => k.label == det.label && _rectIoU(k.box, det.box) > iouThreshold,
    );
    if (!overlaps) kept.add(det);
  }
  return kept;
}

double _rectIoU(cv.Rect a, cv.Rect b) {
  final iw = _ov1D(a.x, a.x + a.width, b.x, b.x + b.width).toDouble();
  final ih = _ov1D(a.y, a.y + a.height, b.y, b.y + b.height).toDouble();
  if (iw <= 0 || ih <= 0) return 0.0;
  final inter = iw * ih;
  final areaA = (a.width * a.height).toDouble();
  final areaB = (b.width * b.height).toDouble();
  final union = areaA + areaB - inter;
  return union <= 0 ? 0.0 : inter / union;
}

List<_Seed> _seedInitialTargets(
  String inputPath,
  double scale,
  int width,
  int height,
  Map<String, _ColorSpec> specs, {
  required int maxTargets,
  required int? expectedTargetCount,
  required _SemanticDetector? semanticDetector,
  required Map<String, List<List<int>>> targetRanges,
  required TargetMode targetMode,
  required bool dualStreamPreprocessing,
  required bool illuminationNormalization,
  void Function(double)? onProgress,
}) {
  if (specs.isEmpty) return const [];

  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) return const [];

  final kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11));
  final seeds = <_Seed>[];
  final downsampling = (scale - 1.0).abs() > 0.001;
  int scanned = 0;

  for (int frameIdx = 0; frameIdx < _targetSeedFrames; frameIdx++) {
    final (ok, raw) = cap.read();
    if (!ok || raw.isEmpty) {
      raw.dispose();
      break;
    }
    scanned++;

    cv.Mat frame;
    if (downsampling) {
      frame = cv.resize(raw, (width, height), interpolation: cv.INTER_AREA);
      raw.dispose();
    } else {
      frame = raw;
    }

    final targetFrame = dualStreamPreprocessing
        ? enhanceTargetsForModel(frame, targetRanges)
        : (illuminationNormalization && targetMode == TargetMode.auto)
        ? applyIlluminationNorm(frame)
        : frame;

    final hsv = cv.cvtColor(targetFrame, cv.COLOR_BGR2HSV);
    final frameColorBlobs = <String, List<_Blob>>{};
    for (final entry in specs.entries) {
      final mask = _colorMask(hsv, entry.value, kernel);
      final blobs = _allBlobs(mask, _minTargetArea)
          .where((b) => _isTargetSeedCandidate(b, width, height))
          .take(10)
          .toList();
      frameColorBlobs[entry.key] = blobs;

      for (final blob in blobs) {
        _upsertSeed(seeds, entry.key, blob, frameIdx);
      }
    }

    if (semanticDetector != null && frameIdx % _semanticScanEvery == 0) {
      final detections = semanticDetector.detect(
        frame,
        threshold: _semanticCupThreshold,
      );
      for (final det in detections.where((d) => d.label == 'target_standing')) {
        final match = _bestColorBlobForSemanticBox(det.box, frameColorBlobs);
        if (match == null) continue;
        final (label, blob) = match;
        final merged = _Blob(
          _unionRect(det.box, blob.box),
          math.max(blob.area, (det.box.width * det.box.height) * 0.55),
          math.max(blob.fillRatio, 0.45),
        );
        if (_isTargetSeedCandidate(merged, width, height)) {
          _upsertSeed(seeds, label, merged, frameIdx, semanticBoost: true);
        }
      }
    }
    hsv.dispose();
    if (!identical(targetFrame, frame)) targetFrame.dispose();
    frame.dispose();
    onProgress?.call(frameIdx / _targetSeedFrames);
  }

  cap.release();
  kernel.dispose();
  if (scanned == 0) return const [];

  final minHits = math.max(_seedMinHits, (scanned * _seedMinHitRatio).round());
  final stable = seeds.where((s) => s.hits >= minHits).toList()
    ..sort((a, b) => b.score.compareTo(a.score));

  final selected = <_Seed>[];
  for (final seed in stable) {
    final duplicate = selected.any((other) {
      if (other.label != seed.label) return false;
      final dist = _centerDistance(seed.box, other.box);
      final gate = math.max(34.0, (seed.box.width + other.box.width) * 0.65);
      return dist < gate || _rectOverlapFrac(seed.box, other.box) > 0.25;
    });
    if (!duplicate) selected.add(seed);
  }

  final hasAssignmentColours = specs.keys.toSet().containsAll(const [
    'red',
    'blue',
    'yellow',
  ]);
  final assignmentExpandedCount = hasAssignmentColours
      ? const ['red', 'blue', 'yellow'].fold<int>(
          0,
          (sum, label) =>
              sum +
              selected
                  .where((s) => s.label == label)
                  .expand(_splitMergedSeedIfNeeded)
                  .length,
        )
      : 0;
  final looksLikeSixTargetScene =
      hasAssignmentColours &&
      selected.length >= 5 &&
      assignmentExpandedCount >= 6;
  final controlledScene =
      hasAssignmentColours &&
      (expectedTargetCount == 6 || looksLikeSixTargetScene);
  final output = <_Seed>[];
  if (controlledScene) {
    for (final label in const ['red', 'blue', 'yellow']) {
      final expanded =
          selected
              .where((s) => s.label == label)
              .expand(_splitMergedSeedIfNeeded)
              .where(
                (s) => _isTargetSeedCandidate(
                  _Blob(s.box, s.area, 0.45),
                  width,
                  height,
                ),
              )
              .toList()
            ..sort((a, b) => _seedRank(b).compareTo(_seedRank(a)));

      final colourSeeds = <_Seed>[];
      for (final seed in expanded) {
        final duplicate = colourSeeds.any((other) {
          final dist = _centerDistance(seed.box, other.box);
          final gate = math.max(
            22.0,
            (seed.box.width + other.box.width) * 0.55,
          );
          return dist < gate || _rectOverlapFrac(seed.box, other.box) > 0.20;
        });
        if (!duplicate) colourSeeds.add(seed);
        if (colourSeeds.length == 2) break;
      }
      output.addAll(colourSeeds);
    }
  } else {
    final limit = (expectedTargetCount ?? maxTargets)
        .clamp(1, _maxTracksPerLabel)
        .toInt();
    final expanded =
        selected
            .expand(_splitMergedSeedIfNeeded)
            .where(
              (s) => _isTargetSeedCandidate(
                _Blob(s.box, s.area, 0.45),
                width,
                height,
              ),
            )
            .toList()
          ..sort((a, b) => _seedRank(b).compareTo(_seedRank(a)));
    final bestRank = expanded.isEmpty ? 0.0 : _seedRank(expanded.first);
    final perLabel = <String, int>{};
    for (final seed in expanded) {
      if ((perLabel[seed.label] ?? 0) >= 2) continue;
      final rank = _seedRank(seed);
      final strongEnough =
          output.isEmpty || rank >= bestRank * 0.22 || seed.hits >= minHits + 2;
      if (!strongEnough) continue;
      final duplicate = output.any((other) {
        final dist = _centerDistance(seed.box, other.box);
        final gate = math.max(24.0, (seed.box.width + other.box.width) * 0.55);
        return dist < gate || _rectOverlapFrac(seed.box, other.box) > 0.22;
      });
      if (duplicate) continue;
      output.add(seed);
      perLabel[seed.label] = (perLabel[seed.label] ?? 0) + 1;
      if (output.length >= limit) break;
    }
  }

  output.sort((a, b) {
    final ax = a.box.x + a.box.width / 2.0;
    final bx = b.box.x + b.box.width / 2.0;
    final x = ax.compareTo(bx);
    if (x != 0) return x;
    final ay = a.box.y + a.box.height / 2.0;
    final by = b.box.y + b.box.height / 2.0;
    return ay.compareTo(by);
  });
  if (targetMode == TargetMode.controlled && output.length < 3) {
    return const [];
  }
  return output;
}

double _seedRank(_Seed seed) {
  final box = seed.box;
  final ratio = box.height / math.max(1, box.width);
  final uprightScore = 1.0 - (ratio - 2.2).abs().clamp(0.0, 2.2) / 2.2;
  final edgePenalty = box.y < 8 ? 0.25 : 1.0;
  return seed.score * (0.75 + uprightScore * 0.35) * edgePenalty;
}

List<_Seed> _splitMergedSeedIfNeeded(_Seed seed) {
  return _splitMergedBlobIfNeeded(_Blob(seed.box, seed.area, 0.5))
      .map(
        (blob) => _Seed(
          label: seed.label,
          box: blob.box,
          area: blob.area,
          lastFrame: seed.lastFrame,
        )..hits = seed.hits,
      )
      .toList();
}

List<_Blob> _splitMergedBlobIfNeeded(_Blob blob) {
  final box = blob.box;
  final ratio = box.height / math.max(1, box.width);
  if (blob.area < 5200 && box.width < 64) return [blob];
  if (ratio > 1.9) {
    final h1 = (box.height * 0.52).round();
    final h2 = box.height - h1;
    return [
      _Blob(
        cv.Rect(box.x, box.y, box.width, h1),
        blob.area * 0.5,
        blob.fillRatio,
      ),
      _Blob(
        cv.Rect(box.x, box.y + h1, box.width, h2),
        blob.area * 0.5,
        blob.fillRatio,
      ),
    ];
  }

  if (ratio < 1.15 && box.width > 100) {
    final w1 = (box.width * 0.50).round();
    final w2 = box.width - w1;
    return [
      _Blob(
        cv.Rect(box.x, box.y, w1, box.height),
        blob.area * 0.5,
        blob.fillRatio,
      ),
      _Blob(
        cv.Rect(box.x + w1, box.y, w2, box.height),
        blob.area * 0.5,
        blob.fillRatio,
      ),
    ];
  }

  return [blob];
}

void _upsertSeed(
  List<_Seed> seeds,
  String label,
  _Blob blob,
  int frameIdx, {
  bool semanticBoost = false,
}) {
  _Seed? best;
  var bestDist = double.infinity;
  for (final seed in seeds.where((s) => s.label == label)) {
    final dist = _centerDistance(seed.box, blob.box);
    final gate = math.max(32.0, (seed.box.width + blob.box.width) * 0.75);
    if ((dist < gate || _rectOverlapFrac(seed.box, blob.box) > 0.25) &&
        dist < bestDist) {
      best = seed;
      bestDist = dist;
    }
  }

  if (best == null) {
    final seed = _Seed(
      label: label,
      box: blob.box,
      area: blob.area,
      lastFrame: frameIdx,
    );
    if (semanticBoost) seed.hits++;
    seeds.add(seed);
  } else {
    best.update(blob, frameIdx);
    if (semanticBoost) best.hits++;
  }
}

(String, _Blob)? _bestColorBlobForSemanticBox(
  cv.Rect box,
  Map<String, List<_Blob>> colorBlobs,
) {
  String? bestLabel;
  _Blob? bestBlob;
  var bestOverlap = 0.0;
  for (final entry in colorBlobs.entries) {
    for (final blob in entry.value) {
      final overlap = _rectOverlapFrac(box, blob.box);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestLabel = entry.key;
        bestBlob = blob;
      }
    }
  }
  if (bestLabel == null || bestBlob == null || bestOverlap < 0.08) {
    return null;
  }
  return (bestLabel, bestBlob);
}

cv.Rect _unionRect(cv.Rect a, cv.Rect b) {
  final x1 = math.min(a.x, b.x);
  final y1 = math.min(a.y, b.y);
  final x2 = math.max(a.x + a.width, b.x + b.width);
  final y2 = math.max(a.y + a.height, b.y + b.height);
  return cv.Rect(x1, y1, math.max(1, x2 - x1), math.max(1, y2 - y1));
}

bool _isTargetSeedCandidate(_Blob blob, int width, int height) {
  final box = blob.box;
  final frameArea = width * height;
  final boxArea = box.width * box.height;
  final minArea = math.max(_minTargetArea, frameArea * _minSeedAreaFraction);
  final ratio = box.height / math.max(1, box.width);
  if (blob.area < minArea) return false;
  if (box.x <= 2 || box.y <= 2) return false;
  if (box.y <= math.max(5, (height * 0.015).round())) return false;
  if (box.x + box.width >= width - 2) return false;
  if (box.y + box.height >= height - 2) return false;
  if (box.width < 10 || box.height < 10) return false;
  if (box.width > width * 0.20 || box.height > height * 0.56) return false;
  if (ratio < 0.75 || ratio > 5.2) return false;
  if (blob.fillRatio < 0.24) return false;
  if (boxArea > frameArea * 0.055) return false;
  if (blob.area > frameArea * 0.040) return false;
  return true;
}

bool _isRuntimeTargetCandidate(_Blob blob, int width, int height) {
  final box = blob.box;
  final frameArea = width * height;
  final boxArea = box.width * box.height;
  if (blob.area < _minTargetArea) return false;
  if (box.width < 8 || box.height < 8) return false;
  if (box.width > width * 0.36 || box.height > height * 0.62) return false;
  if (boxArea > frameArea * 0.11) return false;
  if (blob.fillRatio < 0.10) return false;
  return true;
}

bool _isQualifiedTargetTrack(
  _Track track,
  int width,
  int height,
  double minDisplayConfidence,
) {
  if (track.label == 'mover' || !track.baselineSet) return false;
  final frameArea = width * height;
  final minArea = math.max(_minTargetArea, frameArea * _minSeedAreaFraction);
  if (track.initialArea < minArea) return false;
  if (track.maxArea < minArea) return false;
  if (track.confidence >= minDisplayConfidence) return true;
  return track.fallFrame != null && track.initialArea >= minArea * 1.25;
}

double _centerDistance(cv.Rect a, cv.Rect b) {
  final ax = a.x + a.width / 2.0;
  final ay = a.y + a.height / 2.0;
  final bx = b.x + b.width / 2.0;
  final by = b.y + b.height / 2.0;
  return math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

double _rectOverlapFrac(cv.Rect a, cv.Rect b) {
  final iw = _ov1D(a.x, a.x + a.width, b.x, b.x + b.width).toDouble();
  final ih = _ov1D(a.y, a.y + a.height, b.y, b.y + b.height).toDouble();
  if (iw <= 0 || ih <= 0) return 0.0;
  final inter = iw * ih;
  final smaller = math.min(
    (a.width * a.height).toDouble(),
    (b.width * b.height).toDouble(),
  );
  return smaller <= 0 ? 0.0 : inter / smaller;
}

Map<String, _ColorSpec> _discoverColors(
  String inputPath, {
  void Function(double)? onProgress,
}) {
  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) return {};

  final hueHist = List<int>.filled(180, 0);
  final samples = <List<int>>[];
  int scanned = 0;

  for (int i = 0; i < _discoverFrames; i++) {
    final (ok, raw) = cap.read();
    if (!ok || raw.isEmpty) {
      raw.dispose();
      break;
    }
    scanned++;
    final small = cv.resize(raw, (
      _discoverWidth,
      _discoverHeight,
    ), interpolation: cv.INTER_AREA);
    raw.dispose();
    final hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV);
    small.dispose();

    final h = cv.extractChannel(hsv, 0);
    final s = cv.extractChannel(hsv, 1);
    final v = cv.extractChannel(hsv, 2);
    hsv.dispose();

    final hData = h.data;
    final sData = s.data;
    final vData = v.data;
    for (int p = 0; p < hData.length; p++) {
      final ss = sData[p], vv = vData[p];
      if (ss < _minSaturation || vv < _minValue || vv > _maxValue) continue;
      hueHist[hData[p]]++;
      if (samples.length < 5000) samples.add([hData[p], ss, vv]);
    }
    h.dispose();
    s.dispose();
    v.dispose();
    onProgress?.call(i / _discoverFrames);
  }
  cap.release();
  if (scanned == 0) return {};

  final smoothed = List<double>.filled(180, 0);
  for (int i = 0; i < 180; i++) {
    double sum = 0;
    int cnt = 0;
    for (int j = -3; j <= 3; j++) {
      final idx = (i + j + 180) % 180;
      sum += hueHist[idx];
      cnt++;
    }
    smoothed[i] = sum / cnt;
  }

  final maxVal = smoothed.fold<double>(0, math.max);
  if (maxVal <= 0) return {};
  final minHeight = maxVal * 0.05;

  final peaks = <int>[];
  for (int i = 0; i < 180; i++) {
    final left = smoothed[(i - 1 + 180) % 180];
    final right = smoothed[(i + 1) % 180];
    if (smoothed[i] > left && smoothed[i] > right && smoothed[i] >= minHeight) {
      peaks.add(i);
    }
  }
  peaks.sort((a, b) => smoothed[b].compareTo(smoothed[a]));
  final keepPeaks = <int>[];
  for (final pk in peaks) {
    bool tooClose = false;
    for (final k in keepPeaks) {
      var d = (pk - k).abs();
      if (d > 90) d = 180 - d;
      if (d < _hueMergeDistance) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) keepPeaks.add(pk);
    if (keepPeaks.length >= _maxDiscoveredColors) break;
  }

  final specs = <String, _ColorSpec>{};
  final usedNames = <String>{};
  for (final hue in keepPeaks) {
    int sSum = 0, vSum = 0, n = 0;
    for (final s in samples) {
      var d = (s[0] - hue).abs();
      if (d > 90) d = 180 - d;
      if (d <= _hueWindow) {
        sSum += s[1];
        vSum += s[2];
        n++;
      }
    }
    final sAvg = n > 0 ? (sSum / n).round() : 200;
    final vAvg = n > 0 ? (vSum / n).round() : 200;

    final ranges = <List<int>>[];
    final lo = hue - _hueWindow;
    final hi = hue + _hueWindow;
    final sLo = math.max(_minSaturation, sAvg - 80);
    final vLo = math.max(_minValue, vAvg - 80);
    if (lo < 0) {
      ranges.add([0, hi, sLo, vLo]);
      ranges.add([180 + lo, 179, sLo, vLo]);
    } else if (hi > 179) {
      ranges.add([lo, 179, sLo, vLo]);
      ranges.add([0, hi - 180, sLo, vLo]);
    } else {
      ranges.add([lo, hi, sLo, vLo]);
    }

    final rgb = _hsvToRgb(hue, sAvg, vAvg);
    final packed = (0xFF << 24) | (rgb.$1 << 16) | (rgb.$2 << 8) | rgb.$3;

    var name = _nameForHue(hue);
    var key = name;
    int suffix = 2;
    while (usedNames.contains(key)) {
      key = '$name $suffix';
      suffix++;
    }
    usedNames.add(key);

    specs[key] = _ColorSpec(ranges, packed);
  }

  return specs;
}

String _nameForHue(int hue) {
  if (hue < 8 || hue >= 170) return 'red';
  if (hue < 18) return 'orange';
  if (hue < 32) return 'yellow';
  if (hue < 45) return 'lime';
  if (hue < 75) return 'green';
  if (hue < 95) return 'teal';
  if (hue < 115) return 'cyan';
  if (hue < 135) return 'blue';
  if (hue < 155) return 'purple';
  return 'magenta';
}

(int, int, int) _hsvToRgb(int hOpenCV, int s, int v) {
  final h = hOpenCV * 2.0;
  final ss = s / 255.0;
  final vv = v / 255.0;
  final c = vv * ss;
  final hp = h / 60.0;
  final x = c * (1 - ((hp % 2) - 1).abs());
  double r1 = 0, g1 = 0, b1 = 0;
  if (hp < 1) {
    r1 = c;
    g1 = x;
  } else if (hp < 2) {
    r1 = x;
    g1 = c;
  } else if (hp < 3) {
    g1 = c;
    b1 = x;
  } else if (hp < 4) {
    g1 = x;
    b1 = c;
  } else if (hp < 5) {
    r1 = x;
    b1 = c;
  } else {
    r1 = c;
    b1 = x;
  }
  final m = vv - c;
  return (
    ((r1 + m) * 255).clamp(0, 255).round(),
    ((g1 + m) * 255).clamp(0, 255).round(),
    ((b1 + m) * 255).clamp(0, 255).round(),
  );
}

cv.Scalar _bgrFromPacked(int packedArgb) {
  final r = (packedArgb >> 16) & 0xFF;
  final g = (packedArgb >> 8) & 0xFF;
  final b = packedArgb & 0xFF;
  return cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble());
}

List<String> _confirmActiveTargets(
  String inputPath,
  double scale,
  double confThreshold,
  Map<String, _ColorSpec> discovered, {
  void Function(double)? onProgress,
}) {
  if (discovered.isEmpty) return [];
  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) return discovered.keys.toList();

  final kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5));
  final hits = {for (final k in discovered.keys) k: 0};
  int scanned = 0;
  final downsampling = (scale - 1.0).abs() > 0.001;
  const scanFrames = 120;

  for (int i = 0; i < scanFrames; i++) {
    final (ok, raw) = cap.read();
    if (!ok || raw.isEmpty) {
      raw.dispose();
      break;
    }
    scanned++;
    cv.Mat frame;
    if (downsampling) {
      frame = cv.resize(raw, (
        (raw.cols * scale).round(),
        (raw.rows * scale).round(),
      ), interpolation: cv.INTER_AREA);
      raw.dispose();
    } else {
      frame = raw;
    }
    final hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV);
    for (final entry in discovered.entries) {
      final mask = _colorMask(hsv, entry.value, kernel);
      if (_allBlobs(mask, _minTargetArea).isNotEmpty) {
        hits[entry.key] = hits[entry.key]! + 1;
      }
    }
    frame.dispose();
    hsv.dispose();
    onProgress?.call(i / scanFrames);
  }
  cap.release();
  kernel.dispose();

  final adjustedRatio = _baseDetectRatio * (0.4 + confThreshold * 0.9);
  final threshold = (scanned * adjustedRatio).clamp(2.0, double.infinity);
  return discovered.keys.where((n) => hits[n]! >= threshold).toList();
}

cv.Mat _colorMask(cv.Mat hsv, _ColorSpec spec, cv.Mat kernel) {
  final hCh = cv.extractChannel(hsv, 0);
  final sCh = cv.extractChannel(hsv, 1);
  final vCh = cv.extractChannel(hsv, 2);

  cv.Mat? combined;
  for (final r in spec.ranges) {
    final hLo = r[0], hHi = r[1], sLo = r[2], vLo = r[3];
    final hM = _ch1Range(hCh, hLo, hHi);
    final sM = _ch1Range(sCh, sLo, 255);
    final vM = _ch1Range(vCh, vLo, 255);
    final sv = cv.bitwiseAND(sM, vM);
    final part = cv.bitwiseAND(hM, sv);
    hM.dispose();
    sM.dispose();
    vM.dispose();
    sv.dispose();

    if (combined == null) {
      combined = part;
    } else {
      final merged = cv.bitwiseOR(combined, part);
      combined.dispose();
      part.dispose();
      combined = merged;
    }
  }
  hCh.dispose();
  sCh.dispose();
  vCh.dispose();

  final opened = cv.morphologyEx(combined!, cv.MORPH_OPEN, kernel);
  final closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel);
  combined.dispose();
  opened.dispose();
  return closed;
}

cv.Mat _ch1Range(cv.Mat ch, int lo, int hi) {
  final loM = cv.Mat.fromScalar(
    1,
    1,
    cv.MatType.CV_8UC1,
    cv.Scalar(lo.toDouble(), 0, 0),
  );
  final hiM = cv.Mat.fromScalar(
    1,
    1,
    cv.MatType.CV_8UC1,
    cv.Scalar(hi.toDouble(), 0, 0),
  );
  final r = cv.inRange(ch, loM, hiM);
  loM.dispose();
  hiM.dispose();
  return r;
}

class _Blob {
  final cv.Rect box;
  final double area;
  final double fillRatio;
  final bool semantic;
  _Blob(this.box, this.area, this.fillRatio, {this.semantic = false});
}

List<_Blob> _allBlobs(cv.Mat mask, double minArea) {
  final (contours, hierarchy) = cv.findContours(
    mask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE,
  );
  mask.dispose();
  hierarchy.dispose();
  if (contours.isEmpty) return const [];

  final out = <_Blob>[];
  for (int i = 0; i < contours.length; i++) {
    final area = cv.contourArea(contours[i]);
    if (area < minArea) continue;
    final box = cv.boundingRect(contours[i]);
    final boxArea = box.width * box.height;
    if (boxArea < 1) continue;
    final fill = area / boxArea.toDouble();
    if (fill < _minFillRatio) continue;
    out.add(_Blob(box, area, fill));
  }
  out.sort((a, b) => b.area.compareTo(a.area));
  return out;
}

List<_Blob> _allBlobsRaw(cv.Mat mask, double minArea) {
  final (contours, hierarchy) = cv.findContours(
    mask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE,
  );
  mask.dispose();
  hierarchy.dispose();
  if (contours.isEmpty) return const [];

  final out = <_Blob>[];
  for (int i = 0; i < contours.length; i++) {
    final area = cv.contourArea(contours[i]);
    if (area < minArea) continue;
    final box = cv.boundingRect(contours[i]);
    if (box.width < 1 || box.height < 1) continue;
    out.add(_Blob(box, area, 1.0));
  }
  out.sort((a, b) => b.area.compareTo(a.area));
  return out;
}

bool _overlapsBurnZone(cv.Rect box, List<cv.Rect> zones) {
  if (zones.isEmpty) return false;
  final cx = box.x + box.width / 2.0;
  final cy = box.y + box.height / 2.0;
  for (final z in zones) {
    final left = z.x - _impactBurnPad;
    final top = z.y - _impactBurnPad;
    final right = z.x + z.width + _impactBurnPad;
    final bottom = z.y + z.height + _impactBurnPad;
    if (cx >= left && cx <= right && cy >= top && cy <= bottom) {
      return true;
    }
  }
  return false;
}

void _drawLabel(cv.Mat frame, String text, int x, int y, cv.Scalar color) {
  final ty = (y - 6).clamp(12, frame.rows - 1);
  cv.putText(
    frame,
    text,
    cv.Point(x, ty),
    cv.FONT_HERSHEY_SIMPLEX,
    0.55,
    color,
    thickness: 2,
  );
}
