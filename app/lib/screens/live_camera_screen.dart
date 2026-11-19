import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../pipeline/live_yolo_detector.dart';
import '../theme.dart';
import '../widgets/glass_card.dart';

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _controller;
  LiveYoloDetector? _detector;
  var _initializing = true;
  var _busy = false;
  var _modelReady = false;
  var _message = 'Starting camera...';
  var _lastInferenceMs = 0;
  var _score = 0;
  var _totalTargets = 0;
  final _detections = <LiveDetection>[];
  final _path = <Offset>[];
  final _targetTracks = <_LiveTargetTrack>[];
  final _fallingOrder = <String>[];
  final _motionTracker = _LiveMotionTracker();
  int _lastFrameStartedMs = 0;
  int _nextTrackId = 1;

  int get _pathPointCount => _path.where(_isRealPathPoint).length;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    final controller = _controller;
    if (controller != null && controller.value.isStreamingImages) {
      unawaited(controller.stopImageStream().catchError((_) {}));
    }
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _init() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _initializing = false;
          _message = 'No camera found';
        });
        return;
      }
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      final detector = await LiveYoloDetector.load();
      if (!mounted) {
        await controller.dispose();
        detector?.close();
        return;
      }
      setState(() {
        _controller = controller;
        _detector = detector;
        _modelReady = detector != null;
        _initializing = false;
        _message = detector == null
            ? 'Live camera ready - model asset not available'
            : 'YOLO image stream ready (${detector.backendName})';
      });
      if (detector != null) {
        await controller.startImageStream(_onCameraImage);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _initializing = false;
        _message = 'Camera error: $e';
      });
    }
  }

  void _onCameraImage(CameraImage image) {
    final detector = _detector;
    final controller = _controller;
    final now = DateTime.now().millisecondsSinceEpoch;
    if (_busy ||
        detector == null ||
        controller == null ||
        !controller.value.isInitialized ||
        now - _lastFrameStartedMs < 80) {
      return;
    }
    _busy = true;
    _lastFrameStartedMs = now;
    final sw = Stopwatch()..start();

    try {
      final rgb = _cameraImageToRgbResized(
        image,
        detector.inputWidth,
        detector.inputHeight,
      );
      final rawDets = detector.detectRgb(
        rgb,
        width: detector.inputWidth,
        height: detector.inputHeight,
        threshold: 0.18,
      );
      final dets = _updateTracks(rawDets, rgb);
      if (!mounted) return;
      setState(() {
        _detections
          ..clear()
          ..addAll(dets);
        _lastInferenceMs = sw.elapsedMilliseconds;
        _message =
            '${detector.backendName}: ${dets.where((d) => d.isTarget).length} targets';
      });
    } catch (e) {
      if (mounted) setState(() => _message = 'Live pass skipped: $e');
    } finally {
      _busy = false;
    }
  }

  List<LiveDetection> _updateTracks(
    List<LiveDetection> detections,
    Uint8List rgb,
  ) {
    final targetDets = _dedupeTargetDetections(
      detections.where((d) => d.isTarget).toList()
        ..sort((a, b) => b.score.compareTo(a.score)),
    );

    for (final track in _targetTracks) {
      track.missing++;
    }

    final usedTracks = <_LiveTargetTrack>{};
    for (final det in targetDets) {
      _LiveTargetTrack? best;
      var bestScore = 0.0;
      for (final track in _targetTracks) {
        if (usedTracks.contains(track)) continue;
        final score = _associationScore(track.rect, det.rect);
        if (score > bestScore) {
          bestScore = score;
          best = track;
        }
      }
      if (best == null || bestScore < 0.30) {
        final canSpawn =
            det.score >= 0.30 ||
            (det.label == 'target_fallen' && det.score >= 0.24);
        if (!canSpawn) continue;
        best = _LiveTargetTrack('T${_nextTrackId++}', det.rect);
        _targetTracks.add(best);
      }
      usedTracks.add(best);
      best.rect = _smoothRect(best.rect, det.rect, 0.55);
      best.lastLabel = det.label;
      best.lastScore = det.score;
      best.hits++;
      best.missing = 0;
      if (det.label == 'target_standing') best.seenStanding = true;
      if (det.label == 'target_fallen' && !best.isFallen) {
        best.fallenHits++;
        if (best.fallenHits >= 2) {
          best.isFallen = true;
          _fallingOrder.add(best.id);
        }
      } else if (det.label == 'target_standing') {
        best.fallenHits = math.max(0, best.fallenHits - 1);
      }
    }

    _targetTracks.removeWhere((t) => t.missing > 5 && !t.isConfirmed);
    _targetTracks.removeWhere((t) => t.missing > 12);

    final visibleTargets = _targetTracks
        .where((t) => t.isConfirmed && t.missing <= 5)
        .toList();
    _totalTargets = visibleTargets.length;
    _score = visibleTargets.where((t) => t.isFallen).length;

    final display = <LiveDetection>[
      for (final t in visibleTargets)
        LiveDetection(
          label: t.isFallen ? 'target_fallen' : 'target_standing',
          score: t.lastScore,
          rect: t.rect,
        ),
    ];

    final suppressRects = visibleTargets.map((t) => _expandRect(t.rect, 0.04));
    final motionCenter = _motionTracker.update(
      rgb,
      _detector?.inputWidth ?? 1,
      _detector?.inputHeight ?? 1,
      suppressRects,
    );
    final cars = detections.where((d) => d.isCar).toList()
      ..sort((a, b) => b.score.compareTo(a.score));
    LiveDetection? car;
    var yoloCarLocked = false;
    if (cars.isNotEmpty && cars.first.score >= 0.12) {
      car = cars.first;
      yoloCarLocked = true;
    } else if (motionCenter != null) {
      car = LiveDetection(
        label: 'rc_car',
        score: 0.35,
        rect: _rectAround(motionCenter, 0.15, 0.11),
      );
    }
    if (car != null) {
      _appendCarPath(car.rect.center, forceSnap: yoloCarLocked);
      display.add(car);
    }
    return display;
  }

  void _appendCarPath(Offset center, {bool forceSnap = false}) {
    final last = _lastRealPathPoint();
    if (last != null && (last - center).distance > 0.38) {
      if (!forceSnap) return;
      _path.add(const Offset(double.nan, double.nan));
      _path.add(center);
      _trimPath();
      return;
    }
    final smoothed = last == null
        ? center
        : Offset(
            last.dx * 0.72 + center.dx * 0.28,
            last.dy * 0.72 + center.dy * 0.28,
          );
    _path.add(smoothed);
    _trimPath();
  }

  Offset? _lastRealPathPoint() {
    for (var i = _path.length - 1; i >= 0; i--) {
      final p = _path[i];
      if (_isRealPathPoint(p)) return p;
    }
    return null;
  }

  void _trimPath() {
    if (_path.length > 300) _path.removeAt(0);
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      body: Stack(
        children: [
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFF07110F), Color(0xFF15211C)],
              ),
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  Expanded(
                    flex: 6,
                    child: GlassCard(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Row(
                            children: [
                              IconButton(
                                onPressed: () => Navigator.pop(context),
                                icon: const Icon(Icons.arrow_back_rounded),
                              ),
                              const Text(
                                'Camera Stream',
                                style: TextStyle(
                                  color: textPrimary,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                              const Spacer(),
                              _LiveBadge(
                                text: _modelReady ? 'MODEL ON' : 'PREVIEW',
                                color: _modelReady ? success : warning,
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Expanded(
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(14),
                              child: Container(
                                color: Colors.black,
                                child:
                                    _initializing ||
                                        controller == null ||
                                        !controller.value.isInitialized
                                    ? Center(
                                        child: Text(
                                          _message,
                                          style: const TextStyle(
                                            color: textSecondary,
                                          ),
                                        ),
                                      )
                                    : Stack(
                                        fit: StackFit.expand,
                                        children: [
                                          CameraPreview(controller),
                                          CustomPaint(
                                            painter: _LiveOverlayPainter(
                                              detections: _detections,
                                              path: _path,
                                            ),
                                          ),
                                        ],
                                      ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    flex: 5,
                    child: GlassCard(
                      padding: const EdgeInsets.all(16),
                      child: _buildSidePanel(),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidePanel() {
    final carConf = _detections
        .where((d) => d.isCar)
        .fold<double>(0, (m, d) => math.max(m, d.score));
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'Realtime Metrics',
          style: TextStyle(
            color: textPrimary,
            fontSize: 17,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 14),
        _MetricTile(
          label: 'Score',
          value: '$_score / $_totalTargets',
          icon: Icons.emoji_events_rounded,
        ),
        const SizedBox(height: 10),
        _MetricTile(
          label: 'Visible targets',
          value: '$_totalTargets',
          icon: Icons.filter_center_focus_rounded,
        ),
        const SizedBox(height: 10),
        _MetricTile(
          label: 'RC path points',
          value: '$_pathPointCount',
          icon: Icons.route_rounded,
        ),
        const SizedBox(height: 10),
        _MetricTile(
          label: 'Last live pass',
          value: _lastInferenceMs == 0 ? '-' : '$_lastInferenceMs ms',
          icon: Icons.speed_rounded,
        ),
        const SizedBox(height: 16),
        Text(
          _message,
          style: const TextStyle(color: textSecondary, fontSize: 12),
        ),
        const SizedBox(height: 16),
        const Text(
          'Tracked Objects',
          style: TextStyle(
            color: textPrimary,
            fontSize: 13,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: ListView(
            children: [
              _DetectionRow(
                label: 'RC car',
                value: carConf,
                color: accentSecondary,
              ),
              for (final det in _detections.where((d) => d.isTarget))
                _DetectionRow(
                  label: det.displayLabel,
                  value: det.score,
                  color: det.label == 'target_fallen' ? success : accentPrimary,
                ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Order: ${_fallingOrder.isEmpty ? "-" : _fallingOrder.join(" -> ")}',
          style: const TextStyle(color: textMuted, fontSize: 11),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }
}

bool _isRealPathPoint(Offset point) => point.dx.isFinite && point.dy.isFinite;

class _LiveTargetTrack {
  final String id;
  Rect rect;
  bool seenStanding = false;
  bool isFallen = false;
  int hits = 0;
  int missing = 0;
  int fallenHits = 0;
  double lastScore = 0;
  String lastLabel = 'target_standing';

  _LiveTargetTrack(this.id, this.rect);

  bool get isConfirmed => hits >= 2 || lastScore >= 0.55;
}

class _LiveOverlayPainter extends CustomPainter {
  final List<LiveDetection> detections;
  final List<Offset> path;

  _LiveOverlayPainter({required this.detections, required this.path});

  @override
  void paint(Canvas canvas, Size size) {
    if (path.any(_isRealPathPoint)) {
      final paint = Paint()
        ..color = accentSecondary
        ..strokeWidth = 4
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      Path? currentSegment;
      Offset? singlePoint;
      var segmentPointCount = 0;

      void flushSegment() {
        if (currentSegment == null) return;
        if (segmentPointCount == 1 && singlePoint != null) {
          canvas.drawCircle(
            Offset(singlePoint!.dx * size.width, singlePoint!.dy * size.height),
            3,
            Paint()..color = accentSecondary,
          );
        } else {
          canvas.drawPath(currentSegment!, paint);
        }
        currentSegment = null;
        singlePoint = null;
        segmentPointCount = 0;
      }

      for (final point in path) {
        if (!_isRealPathPoint(point)) {
          flushSegment();
          continue;
        }
        final canvasPoint = Offset(
          point.dx * size.width,
          point.dy * size.height,
        );
        if (currentSegment == null) {
          currentSegment = Path()..moveTo(canvasPoint.dx, canvasPoint.dy);
          singlePoint = point;
          segmentPointCount = 1;
        } else {
          currentSegment!.lineTo(canvasPoint.dx, canvasPoint.dy);
          segmentPointCount++;
        }
      }
      flushSegment();
    }

    for (final det in detections) {
      final color = det.isCar
          ? accentSecondary
          : det.label == 'target_fallen'
          ? success
          : accentPrimary;
      final rect = Rect.fromLTRB(
        det.rect.left * size.width,
        det.rect.top * size.height,
        det.rect.right * size.width,
        det.rect.bottom * size.height,
      );
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = det.isCar ? 3 : 2;
      canvas.drawRect(rect, paint);
      final tp = TextPainter(
        text: TextSpan(
          text: '${det.displayLabel} ${(det.score * 100).round()}%',
          style: TextStyle(
            color: color,
            fontSize: 12,
            fontWeight: FontWeight.w800,
            backgroundColor: Colors.black.withValues(alpha: 0.65),
          ),
        ),
        textDirection: ui.TextDirection.ltr,
      )..layout(maxWidth: 160);
      tp.paint(canvas, Offset(rect.left, math.max(2, rect.top - 18)));
    }
  }

  @override
  bool shouldRepaint(covariant _LiveOverlayPainter oldDelegate) =>
      oldDelegate.detections != detections || oldDelegate.path != path;
}

class _LiveBadge extends StatelessWidget {
  final String text;
  final Color color;

  const _LiveBadge({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(
        text,
        style: TextStyle(
          color: color,
          fontSize: 11,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _MetricTile({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.035),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        children: [
          Icon(icon, color: accentSecondary, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(color: textMuted, fontSize: 11),
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _DetectionRow extends StatelessWidget {
  final String label;
  final double value;
  final Color color;

  const _DetectionRow({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final v = value.clamp(0.0, 1.0);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          SizedBox(
            width: 92,
            child: Text(
              label,
              style: const TextStyle(color: textSecondary, fontSize: 12),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                minHeight: 7,
                value: v,
                backgroundColor: Colors.white.withValues(alpha: 0.06),
                valueColor: AlwaysStoppedAnimation(color),
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            child: Text(
              '${(v * 100).round()}%',
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: textPrimary,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

Uint8List _cameraImageToRgbResized(CameraImage image, int outW, int outH) {
  final out = Uint8List(outW * outH * 3);
  final srcW = image.width;
  final srcH = image.height;

  if (image.planes.length == 1) {
    final plane = image.planes.first;
    final rowStride = plane.bytesPerRow;
    final pixelStride = plane.bytesPerPixel ?? 4;
    for (var y = 0; y < outH; y++) {
      final sy = (y * srcH / outH).floor().clamp(0, srcH - 1).toInt();
      for (var x = 0; x < outW; x++) {
        final sx = (x * srcW / outW).floor().clamp(0, srcW - 1).toInt();
        final si = sy * rowStride + sx * pixelStride;
        final di = (y * outW + x) * 3;
        out[di] =
            plane.bytes[(si + 2).clamp(0, plane.bytes.length - 1).toInt()];
        out[di + 1] =
            plane.bytes[(si + 1).clamp(0, plane.bytes.length - 1).toInt()];
        out[di + 2] = plane.bytes[si.clamp(0, plane.bytes.length - 1).toInt()];
      }
    }
    return out;
  }

  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];
  final yRowStride = yPlane.bytesPerRow;
  final uvRowStride = uPlane.bytesPerRow;
  final uvPixelStride = uPlane.bytesPerPixel ?? 1;

  for (var y = 0; y < outH; y++) {
    final sy = (y * srcH / outH).floor().clamp(0, srcH - 1).toInt();
    for (var x = 0; x < outW; x++) {
      final sx = (x * srcW / outW).floor().clamp(0, srcW - 1).toInt();
      final yIndex = sy * yRowStride + sx;
      final uvIndex = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
      final yy = yPlane.bytes[yIndex.clamp(0, yPlane.bytes.length - 1).toInt()];
      final uu =
          uPlane.bytes[uvIndex.clamp(0, uPlane.bytes.length - 1).toInt()];
      final vv =
          vPlane.bytes[uvIndex.clamp(0, vPlane.bytes.length - 1).toInt()];
      final yf = yy.toDouble();
      final uf = uu - 128.0;
      final vf = vv - 128.0;
      final r = (yf + 1.402 * vf).round().clamp(0, 255).toInt();
      final g = (yf - 0.344136 * uf - 0.714136 * vf)
          .round()
          .clamp(0, 255)
          .toInt();
      final b = (yf + 1.772 * uf).round().clamp(0, 255).toInt();
      final di = (y * outW + x) * 3;
      out[di] = r;
      out[di + 1] = g;
      out[di + 2] = b;
    }
  }
  return out;
}

class _LiveMotionTracker {
  static const _gridW = 48;
  static const _gridH = 32;
  static const _sampleStep = 4;
  Uint8List? _previousRgb;

  Offset? update(
    Uint8List rgb,
    int width,
    int height,
    Iterable<Rect> suppressRects,
  ) {
    final prev = _previousRgb;
    _previousRgb = Uint8List.fromList(rgb);
    if (prev == null ||
        prev.length != rgb.length ||
        width <= 0 ||
        height <= 0) {
      return null;
    }

    final weights = List<int>.filled(_gridW * _gridH, 0);
    for (var y = 0; y < height; y += _sampleStep) {
      final yn = y / height;
      for (var x = 0; x < width; x += _sampleStep) {
        final xn = x / width;
        if (_insideAnyRect(xn, yn, suppressRects)) continue;
        final i = (y * width + x) * 3;
        final lum = (rgb[i] * 0.299 + rgb[i + 1] * 0.587 + rgb[i + 2] * 0.114)
            .round();
        final prevLum =
            (prev[i] * 0.299 + prev[i + 1] * 0.587 + prev[i + 2] * 0.114)
                .round();
        if ((lum - prevLum).abs() < 24) continue;
        final gx = (x * _gridW ~/ width).clamp(0, _gridW - 1).toInt();
        final gy = (y * _gridH ~/ height).clamp(0, _gridH - 1).toInt();
        weights[gy * _gridW + gx]++;
      }
    }

    final active = List<bool>.filled(_gridW * _gridH, false);
    var activeCount = 0;
    for (var i = 0; i < weights.length; i++) {
      if (weights[i] < 2) continue;
      active[i] = true;
      activeCount++;
    }
    if (activeCount < 4 || activeCount > weights.length * 0.25) return null;

    final visited = List<bool>.filled(weights.length, false);
    var bestWeight = 0;
    var bestX = 0.0;
    var bestY = 0.0;
    for (var i = 0; i < active.length; i++) {
      if (!active[i] || visited[i]) continue;
      final queue = <int>[i];
      visited[i] = true;
      var head = 0;
      var sumW = 0;
      var sumX = 0.0;
      var sumY = 0.0;
      while (head < queue.length) {
        final idx = queue[head++];
        final w = weights[idx];
        final gx = idx % _gridW;
        final gy = idx ~/ _gridW;
        sumW += w;
        sumX += (gx + 0.5) * w;
        sumY += (gy + 0.5) * w;
        for (final n in _neighbors(idx)) {
          if (visited[n] || !active[n]) continue;
          visited[n] = true;
          queue.add(n);
        }
      }
      if (sumW > bestWeight) {
        bestWeight = sumW;
        bestX = sumX / sumW / _gridW;
        bestY = sumY / sumW / _gridH;
      }
    }
    if (bestWeight < 8) return null;
    return Offset(
      bestX.clamp(0.0, 1.0).toDouble(),
      bestY.clamp(0.0, 1.0).toDouble(),
    );
  }

  Iterable<int> _neighbors(int idx) sync* {
    final x = idx % _gridW;
    final y = idx ~/ _gridW;
    if (x > 0) yield idx - 1;
    if (x < _gridW - 1) yield idx + 1;
    if (y > 0) yield idx - _gridW;
    if (y < _gridH - 1) yield idx + _gridW;
  }
}

List<LiveDetection> _dedupeTargetDetections(List<LiveDetection> detections) {
  final kept = <LiveDetection>[];
  for (final det in detections) {
    final overlaps = kept.any((k) => _iou(k.rect, det.rect) > 0.42);
    if (!overlaps) kept.add(det);
  }
  return kept;
}

double _associationScore(Rect track, Rect det) {
  final iou = _iou(track, det);
  final dist = (track.center - det.center).distance;
  final distScore = (1.0 - dist / 0.22).clamp(0.0, 1.0).toDouble();
  final sizeRatio =
      math.min(track.width * track.height, det.width * det.height) /
      math.max(track.width * track.height, det.width * det.height);
  return iou * 0.58 +
      distScore * 0.32 +
      sizeRatio.clamp(0.0, 1.0).toDouble() * 0.10;
}

Rect _smoothRect(Rect oldRect, Rect newRect, double newWeight) {
  final oldWeight = 1.0 - newWeight;
  return Rect.fromLTRB(
    oldRect.left * oldWeight + newRect.left * newWeight,
    oldRect.top * oldWeight + newRect.top * newWeight,
    oldRect.right * oldWeight + newRect.right * newWeight,
    oldRect.bottom * oldWeight + newRect.bottom * newWeight,
  );
}

Rect _expandRect(Rect rect, double pad) {
  return Rect.fromLTRB(
    (rect.left - pad).clamp(0.0, 1.0).toDouble(),
    (rect.top - pad).clamp(0.0, 1.0).toDouble(),
    (rect.right + pad).clamp(0.0, 1.0).toDouble(),
    (rect.bottom + pad).clamp(0.0, 1.0).toDouble(),
  );
}

Rect _rectAround(Offset center, double width, double height) {
  return Rect.fromLTRB(
    (center.dx - width / 2).clamp(0.0, 1.0).toDouble(),
    (center.dy - height / 2).clamp(0.0, 1.0).toDouble(),
    (center.dx + width / 2).clamp(0.0, 1.0).toDouble(),
    (center.dy + height / 2).clamp(0.0, 1.0).toDouble(),
  );
}

bool _insideAnyRect(double x, double y, Iterable<Rect> rects) {
  for (final rect in rects) {
    if (x >= rect.left &&
        x <= rect.right &&
        y >= rect.top &&
        y <= rect.bottom) {
      return true;
    }
  }
  return false;
}

double _iou(Rect a, Rect b) {
  final left = math.max(a.left, b.left);
  final top = math.max(a.top, b.top);
  final right = math.min(a.right, b.right);
  final bottom = math.min(a.bottom, b.bottom);
  if (right <= left || bottom <= top) return 0.0;
  final inter = (right - left) * (bottom - top);
  final union = a.width * a.height + b.width * b.height - inter;
  return union <= 0 ? 0.0 : inter / union;
}
