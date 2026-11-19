enum TargetMode { controlled, auto }

class PipelineOptions {
  final double confidenceThreshold;

  final int? downscaleHeight;

  final int maxTargets;

  final double minDisplayConfidence;

  final bool debugMode;

  final int? expectedTargetCount;

  final TargetMode targetMode;

  final bool illuminationNormalization;

  final bool dualStreamPreprocessing;

  final bool semanticAssist;

  final int frameStep;

  final bool writeAnnotatedVideo;

  final bool earlyStopOnComplete;

  const PipelineOptions({
    this.confidenceThreshold = 0.7,
    this.downscaleHeight = 360,
    this.maxTargets = 6,
    this.minDisplayConfidence = 0.55,
    this.debugMode = false,
    this.expectedTargetCount = 6,
    this.targetMode = TargetMode.controlled,
    this.illuminationNormalization = false,
    this.dualStreamPreprocessing = true,
    this.semanticAssist = false,
    this.frameStep = 2,
    this.writeAnnotatedVideo = true,
    this.earlyStopOnComplete = false,
  });

  const PipelineOptions.mobileFast()
    : confidenceThreshold = 0.7,
      downscaleHeight = 360,
      maxTargets = 6,
      minDisplayConfidence = 0.55,
      debugMode = false,
      expectedTargetCount = null,
      targetMode = TargetMode.controlled,
      illuminationNormalization = false,
      dualStreamPreprocessing = true,
      semanticAssist = false,
      frameStep = 8,
      writeAnnotatedVideo = true,
      earlyStopOnComplete = true;

  const PipelineOptions.v2Hybrid()
    : confidenceThreshold = 0.7,
      downscaleHeight = 360,
      maxTargets = 6,
      minDisplayConfidence = 0.55,
      debugMode = false,
      expectedTargetCount = null,
      targetMode = TargetMode.controlled,
      illuminationNormalization = false,
      dualStreamPreprocessing = true,
      semanticAssist = true,
      frameStep = 3,
      writeAnnotatedVideo = true,
      earlyStopOnComplete = true;

  Map<String, dynamic> toJson() => {
    'confidence_threshold': confidenceThreshold,
    'downscale_height': downscaleHeight,
    'max_targets': maxTargets,
    'min_display_confidence': minDisplayConfidence,
    'debug_mode': debugMode,
    'expected_target_count': expectedTargetCount,
    'target_mode': targetMode.name,
    'illumination_normalization': illuminationNormalization,
    'dual_stream_preprocessing': dualStreamPreprocessing,
    'semantic_assist': semanticAssist,
    'frame_step': frameStep,
    'write_annotated_video': writeAnnotatedVideo,
    'early_stop_on_complete': earlyStopOnComplete,
  };
}

enum PipelineStage { load, detect, classify, annotate, complete }

class PipelineProgress {
  final PipelineStage stage;
  final double progress;
  final String message;
  const PipelineProgress(this.stage, this.progress, this.message);

  static PipelineProgress fromMap(Map m) => PipelineProgress(
    PipelineStage.values.firstWhere((s) => s.name == m['stage']),
    (m['progress'] as num).toDouble(),
    m['message'] as String,
  );
}

class LogEvent {
  final int frame;
  final double tSec;
  final String message;
  final String level;
  final String? color;

  const LogEvent({
    required this.frame,
    required this.tSec,
    required this.message,
    required this.level,
    this.color,
  });

  static LogEvent fromMap(Map m) => LogEvent(
    frame: m['frame'] as int,
    tSec: (m['t'] as num).toDouble(),
    message: m['msg'] as String,
    level: m['level'] as String,
    color: m['color'] as String?,
  );

  Map<String, dynamic> toJson() => {
    'frame': frame,
    't': tSec,
    'msg': message,
    'level': level,
    if (color != null) 'color': color,
  };
}

class CountSample {
  final double tSec;
  final Map<String, int> counts;

  const CountSample(this.tSec, this.counts);

  static CountSample fromMap(Map m) => CountSample(
    (m['t'] as num).toDouble(),
    Map<String, int>.from(m['counts'] as Map),
  );

  Map<String, dynamic> toJson() => {'t': tSec, 'counts': counts};
}

class FallEvent {
  final int x;
  final int y;
  final int boxX;
  final int boxY;
  final int boxW;
  final int boxH;
  final int frame;
  final double tSec;
  final String signal;

  const FallEvent({
    required this.x,
    required this.y,
    required this.boxX,
    required this.boxY,
    required this.boxW,
    required this.boxH,
    required this.frame,
    required this.tSec,
    required this.signal,
  });

  static FallEvent fromMap(Map m) => FallEvent(
    x: m['x'] as int,
    y: m['y'] as int,
    boxX: m['box_x'] as int,
    boxY: m['box_y'] as int,
    boxW: m['box_w'] as int,
    boxH: m['box_h'] as int,
    frame: m['frame'] as int,
    tSec: (m['t'] as num).toDouble(),
    signal: m['signal'] as String,
  );

  Map<String, dynamic> toJson() => {
    'x': x,
    'y': y,
    'box_x': boxX,
    'box_y': boxY,
    'box_w': boxW,
    'box_h': boxH,
    'frame': frame,
    't': tSec,
    'signal': signal,
  };
}

class PipelineResult {
  final int score;
  final int totalTargets;
  final List<String> activeTargets;
  final List<String> fallingOrder;
  final Map<String, double> fallTimesSec;

  final Map<String, double> confidences;

  final List<CountSample> countTimeline;

  final List<LogEvent> eventLog;

  final Map<String, FallEvent> fallEvents;

  final Map<String, int> targetColors;

  final int frameWidth;
  final int frameHeight;

  final int originalWidth;
  final int originalHeight;

  final int carPathPoints;
  final bool pathDrawn;
  final int framesProcessed;
  final double fps;
  final String outputVideoPath;
  final Duration processingTime;
  final PipelineOptions options;
  final Map<String, dynamic> preprocessing;

  const PipelineResult({
    required this.score,
    required this.totalTargets,
    required this.activeTargets,
    required this.fallingOrder,
    required this.fallTimesSec,
    required this.confidences,
    required this.countTimeline,
    required this.eventLog,
    required this.fallEvents,
    required this.targetColors,
    required this.frameWidth,
    required this.frameHeight,
    required this.originalWidth,
    required this.originalHeight,
    required this.carPathPoints,
    required this.pathDrawn,
    required this.framesProcessed,
    required this.fps,
    required this.outputVideoPath,
    required this.processingTime,
    required this.options,
    required this.preprocessing,
  });

  String get fallingOrderString {
    if (fallingOrder.isEmpty) return '(none fell)';
    return fallingOrder.map(_prettyObjectName).join(' -> ');
  }

  String get processingTimeString {
    final s = processingTime.inSeconds;
    if (s < 60) return '${s}s';
    return '${s ~/ 60}m ${s % 60}s';
  }

  static String _prettyObjectName(String id) {
    if (id == 'rc_car') return 'RC Car';
    if (id.startsWith('target_')) {
      final n = id.substring('target_'.length);
      return 'Target $n';
    }
    return id.replaceAll('_', ' ');
  }

  Map<String, dynamic> toJson() => {
    'score': score,
    'total_targets': totalTargets,
    'active_targets': activeTargets,
    'falling_order': fallingOrder,
    'fall_times_sec': fallTimesSec,
    'confidences': confidences,
    'count_timeline': countTimeline.map((s) => s.toJson()).toList(),
    'event_log': eventLog.map((e) => e.toJson()).toList(),
    'fall_events': fallEvents.map((k, v) => MapEntry(k, v.toJson())),
    'target_colors': targetColors,
    'frame_width': frameWidth,
    'frame_height': frameHeight,
    'original_width': originalWidth,
    'original_height': originalHeight,
    'car_path_points': carPathPoints,
    'motion_path_points': carPathPoints,
    'path_drawn': pathDrawn,
    'frames_processed': framesProcessed,
    'fps': fps,
    'output_video': outputVideoPath,
    'processing_time_ms': processingTime.inMilliseconds,
    'options': options.toJson(),
    'preprocessing': preprocessing,
  };

  static PipelineResult fromMap(Map m, PipelineOptions opts) {
    return PipelineResult(
      score: m['score'] as int,
      totalTargets: m['total_targets'] as int,
      activeTargets: List<String>.from(m['active_targets'] as List),
      fallingOrder: List<String>.from(m['falling_order'] as List),
      fallTimesSec: Map<String, double>.from(
        (m['fall_times_sec'] as Map).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
      ),
      confidences: Map<String, double>.from(
        (m['confidences'] as Map).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
      ),
      countTimeline: ((m['count_timeline'] as List).cast<Map>())
          .map(CountSample.fromMap)
          .toList(),
      eventLog: ((m['event_log'] as List).cast<Map>())
          .map(LogEvent.fromMap)
          .toList(),
      fallEvents: () {
        final raw = m['fall_events'];
        if (raw is! Map) return <String, FallEvent>{};
        return Map<String, FallEvent>.from(
          raw.map((k, v) => MapEntry(k as String, FallEvent.fromMap(v as Map))),
        );
      }(),
      targetColors: Map<String, int>.from(
        (m['target_colors'] as Map).map((k, v) => MapEntry(k, v as int)),
      ),
      frameWidth: m['frame_width'] as int,
      frameHeight: m['frame_height'] as int,
      originalWidth: (m['original_width'] ?? m['frame_width']) as int,
      originalHeight: (m['original_height'] ?? m['frame_height']) as int,
      carPathPoints: m['car_path_points'] as int,
      pathDrawn: m['path_drawn'] as bool,
      framesProcessed: m['frames_processed'] as int,
      fps: (m['fps'] as num).toDouble(),
      outputVideoPath: m['output_video'] as String,
      processingTime: Duration(milliseconds: m['processing_time_ms'] as int),
      options: opts,
      preprocessing: Map<String, dynamic>.from(
        (m['preprocessing'] as Map?) ?? const {},
      ),
    );
  }
}
