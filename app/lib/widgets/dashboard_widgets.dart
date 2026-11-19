import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../models/pipeline_result.dart';
import '../theme.dart';

Color resolveColor(String name, Map<String, int> targetColors) {
  final packed = targetColors[name];
  if (packed != null) return Color(packed);
  if (name == 'rc_car') return accentSecondary;
  final idx = name.hashCode.abs() % spectrumColors.length;
  return spectrumColors[idx];
}

String displayName(String name) {
  if (name == 'rc_car') return 'RC Car';
  if (name.startsWith('target_')) {
    return 'Target ${name.substring('target_'.length)}';
  }
  final cleaned = name.replaceAll('_', ' ');
  return cleaned.isEmpty
      ? name
      : '${cleaned[0].toUpperCase()}${cleaned.substring(1)}';
}

class SectionHeader extends StatelessWidget {
  const SectionHeader({super.key, required this.title, this.trailing});
  final String title;
  final String? trailing;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Text(
            title,
            style: const TextStyle(
              color: textPrimary,
              fontSize: 14,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.2,
            ),
          ),
          const Spacer(),
          if (trailing != null)
            Text(
              trailing!,
              style: const TextStyle(
                color: textMuted,
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
            ),
        ],
      ),
    );
  }
}

class ConfidenceBar extends StatelessWidget {
  const ConfidenceBar({
    super.key,
    required this.label,
    required this.value,
    required this.color,
  });
  final String label;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final pct = (value * 100).clamp(0, 100);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(
              label,
              style: const TextStyle(color: textSecondary, fontSize: 12),
            ),
          ),
          Expanded(
            child: Stack(
              children: [
                Container(
                  height: 8,
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: value.clamp(0.0, 1.0),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    height: 8,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [color.withValues(alpha: 0.65), color],
                      ),
                      borderRadius: BorderRadius.circular(4),
                      boxShadow: [
                        BoxShadow(
                          color: color.withValues(alpha: 0.4),
                          blurRadius: 6,
                          offset: const Offset(0, 1),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            child: Text(
              '${pct.toStringAsFixed(1)}%',
              textAlign: TextAlign.right,
              style: const TextStyle(
                color: textPrimary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CountChart extends StatelessWidget {
  const CountChart({super.key, required this.timeline, required this.colors});
  final List<CountSample> timeline;
  final Map<String, Color> colors;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 90,
      child: timeline.isEmpty
          ? const Center(
              child: Text(
                'No data',
                style: TextStyle(color: textMuted, fontSize: 11),
              ),
            )
          : CustomPaint(
              painter: _CountChartPainter(timeline, colors),
              size: Size.infinite,
            ),
    );
  }
}

class _CountChartPainter extends CustomPainter {
  _CountChartPainter(this.samples, this.colors);
  final List<CountSample> samples;
  final Map<String, Color> colors;

  @override
  void paint(Canvas canvas, Size size) {
    if (samples.isEmpty) return;

    final axisPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.06)
      ..strokeWidth = 1;
    for (int g = 0; g < 5; g++) {
      final y = size.height * g / 4;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), axisPaint);
    }

    int maxCount = 1;
    for (final s in samples) {
      for (final v in s.counts.values) {
        if (v > maxCount) maxCount = v;
      }
    }
    maxCount = math.max(maxCount, 2);

    final names = colors.keys.toList();
    final tMin = samples.first.tSec;
    final tMax = samples.last.tSec;
    final dur = math.max(tMax - tMin, 0.001);

    for (final name in names) {
      final path = Path();
      bool first = true;
      for (final s in samples) {
        final x = ((s.tSec - tMin) / dur) * size.width;
        final c = (s.counts[name] ?? 0).toDouble();
        final y = size.height - (c / maxCount) * size.height;
        if (first) {
          path.moveTo(x, y);
          first = false;
        } else {
          path.lineTo(x, y);
        }
      }
      final paint = Paint()
        ..color = colors[name]!.withValues(alpha: 0.85)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round;
      canvas.drawPath(path, paint);
    }

    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (int g = 0; g <= 4; g++) {
      final value = (maxCount * (4 - g) / 4).round();
      tp.text = TextSpan(
        text: '$value',
        style: const TextStyle(color: textMuted, fontSize: 9),
      );
      tp.layout();
      tp.paint(canvas, Offset(2, size.height * g / 4 - 5));
    }
  }

  @override
  bool shouldRepaint(_CountChartPainter old) =>
      old.samples != samples || old.colors != colors;
}

class DetectionLog extends StatelessWidget {
  const DetectionLog({
    super.key,
    required this.events,
    required this.targetColors,
    this.maxItems = 200,
  });
  final List<LogEvent> events;
  final Map<String, int> targetColors;
  final int maxItems;

  @override
  Widget build(BuildContext context) {
    final shown = events.length > maxItems
        ? events.sublist(events.length - maxItems)
        : events;
    return ListView.builder(
      reverse: true,
      itemCount: shown.length,
      padding: EdgeInsets.zero,
      itemBuilder: (_, i) {
        final ev = shown[shown.length - 1 - i];
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 3),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              SizedBox(
                width: 36,
                child: Text(
                  '${ev.tSec.toStringAsFixed(1)}s',
                  style: const TextStyle(color: textMuted, fontSize: 10),
                ),
              ),
              Expanded(
                child: Text(
                  ev.message,
                  style: const TextStyle(color: textPrimary, fontSize: 11),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (ev.color != null) ...[
                const SizedBox(width: 6),
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: resolveColor(ev.color!, targetColors),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ] else
                Icon(
                  _iconForLevel(ev.level),
                  size: 12,
                  color: _colorForLevel(ev.level),
                ),
            ],
          ),
        );
      },
    );
  }

  IconData _iconForLevel(String l) {
    switch (l) {
      case 'success':
        return Icons.check_circle_rounded;
      case 'warn':
        return Icons.warning_rounded;
      case 'event':
        return Icons.bolt_rounded;
      default:
        return Icons.info_rounded;
    }
  }

  Color _colorForLevel(String l) {
    switch (l) {
      case 'success':
        return success;
      case 'warn':
        return warning;
      case 'event':
        return accentPrimary;
      default:
        return accentBlue;
    }
  }
}

class ColorChipRow extends StatelessWidget {
  const ColorChipRow({
    super.key,
    required this.sequence,
    required this.targetColors,
  });
  final List<String> sequence;
  final Map<String, int> targetColors;

  @override
  Widget build(BuildContext context) {
    if (sequence.isEmpty) {
      return const Text('—', style: TextStyle(color: textMuted, fontSize: 12));
    }
    return Wrap(
      spacing: 4,
      runSpacing: 4,
      children: [
        for (final c in sequence)
          () {
            final col = resolveColor(c, targetColors);
            return Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                color: col,
                borderRadius: BorderRadius.circular(4),
                boxShadow: [
                  BoxShadow(
                    color: col.withValues(alpha: 0.4),
                    blurRadius: 4,
                    offset: const Offset(0, 1),
                  ),
                ],
              ),
              child: Center(
                child: Text(
                  _chipText(c),
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            );
          }(),
      ],
    );
  }
}

String _chipText(String id) {
  if (id.startsWith('target_')) return id.substring('target_'.length);
  if (id == 'rc_car') return 'C';
  return id.isEmpty ? '?' : id[0].toUpperCase();
}

class EventTimeline extends StatelessWidget {
  const EventTimeline({
    super.key,
    required this.fallTimes,
    required this.totalDuration,
    required this.position,
    required this.onSeek,
    required this.targetColors,
  });
  final Map<String, double> fallTimes;
  final Duration totalDuration;
  final Duration position;
  final ValueChanged<Duration> onSeek;
  final Map<String, int> targetColors;

  @override
  Widget build(BuildContext context) {
    final dur = totalDuration.inMilliseconds.toDouble();
    final pos = position.inMilliseconds.toDouble().clamp(0.0, dur);
    return LayoutBuilder(
      builder: (context, constraints) {
        const horizontalPad = 12.0;
        final usableWidth = (constraints.maxWidth - horizontalPad * 2).clamp(
          0.0,
          double.infinity,
        );
        return SizedBox(
          height: 56,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              if (dur > 0)
                for (final entry in fallTimes.entries)
                  Builder(
                    builder: (_) {
                      final ratio = (entry.value * 1000.0 / dur).clamp(
                        0.0,
                        1.0,
                      );
                      final color = resolveColor(entry.key, targetColors);
                      return Positioned(
                        left: horizontalPad + ratio * usableWidth - 5,
                        top: 6,
                        child: _FallMarker(color: color),
                      );
                    },
                  ),

              Positioned.fill(
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    trackHeight: 4,
                    activeTrackColor: accentPrimary,
                    inactiveTrackColor: Colors.white.withValues(alpha: 0.08),
                    thumbShape: const RoundSliderThumbShape(
                      enabledThumbRadius: 6,
                    ),
                    overlayShape: SliderComponentShape.noOverlay,
                    thumbColor: Colors.white,
                  ),
                  child: Slider(
                    value: dur > 0 ? pos / dur : 0,
                    onChanged: dur > 0
                        ? (v) =>
                              onSeek(Duration(milliseconds: (v * dur).round()))
                        : null,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _FallMarker extends StatelessWidget {
  const _FallMarker({required this.color});
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: color.withValues(alpha: 0.55),
                blurRadius: 5,
                offset: const Offset(0, 1),
              ),
            ],
          ),
        ),
        Container(width: 2, height: 14, color: color.withValues(alpha: 0.65)),
      ],
    );
  }
}

class FloorMapPanel extends StatelessWidget {
  const FloorMapPanel({
    super.key,
    required this.fallEvents,
    required this.targetColors,
    required this.originalWidth,
    required this.originalHeight,
  });
  final Map<String, FallEvent> fallEvents;
  final Map<String, int> targetColors;
  final int originalWidth;
  final int originalHeight;

  @override
  Widget build(BuildContext context) {
    if (originalWidth <= 0 || originalHeight <= 0) {
      return const SizedBox.shrink();
    }
    return AspectRatio(
      aspectRatio: originalWidth / originalHeight,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.25),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: CustomPaint(
            painter: _FloorMapPainter(
              fallEvents: fallEvents,
              targetColors: targetColors,
              originalWidth: originalWidth,
              originalHeight: originalHeight,
            ),
            size: Size.infinite,
          ),
        ),
      ),
    );
  }
}

class _FloorMapPainter extends CustomPainter {
  _FloorMapPainter({
    required this.fallEvents,
    required this.targetColors,
    required this.originalWidth,
    required this.originalHeight,
  });
  final Map<String, FallEvent> fallEvents;
  final Map<String, int> targetColors;
  final int originalWidth;
  final int originalHeight;

  @override
  void paint(Canvas canvas, Size size) {
    final grid = Paint()
      ..color = Colors.white.withValues(alpha: 0.05)
      ..strokeWidth = 0.6;
    for (int g = 1; g < 4; g++) {
      final gx = size.width * g / 4;
      final gy = size.height * g / 4;
      canvas.drawLine(Offset(gx, 0), Offset(gx, size.height), grid);
      canvas.drawLine(Offset(0, gy), Offset(size.width, gy), grid);
    }

    for (final entry in fallEvents.entries) {
      final ev = entry.value;
      final cx = (ev.x / originalWidth) * size.width;
      final cy = (ev.y / originalHeight) * size.height;
      final color = resolveColor(entry.key, targetColors);
      const arm = 7.0;

      final stroke = Paint()
        ..color = color
        ..strokeWidth = 2.5
        ..strokeCap = StrokeCap.round;
      canvas.drawLine(
        Offset(cx - arm, cy - arm),
        Offset(cx + arm, cy + arm),
        stroke,
      );
      canvas.drawLine(
        Offset(cx - arm, cy + arm),
        Offset(cx + arm, cy - arm),
        stroke,
      );
      canvas.drawCircle(Offset(cx, cy), 2.5, Paint()..color = color);

      final label = _chipText(entry.key);
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 9,
            fontWeight: FontWeight.w700,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(cx + arm + 1, cy - arm - 2));
    }
  }

  @override
  bool shouldRepaint(_FloorMapPainter old) =>
      old.fallEvents != fallEvents ||
      old.targetColors != targetColors ||
      old.originalWidth != originalWidth ||
      old.originalHeight != originalHeight;
}
