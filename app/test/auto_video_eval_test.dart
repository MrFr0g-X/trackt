import 'dart:convert';
import 'dart:io';

import 'package:dsai352_cv/models/pipeline_result.dart';
import 'package:dsai352_cv/pipeline/cv_pipeline.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

const _runAutoEval = bool.fromEnvironment('TRACKT_EVAL_AUTO_VIDEOS');

void main() {
  test('auto target-discovery on real captured videos', () async {
    if (!_runAutoEval) {
      markTestSkipped(
        'Set --dart-define=TRACKT_EVAL_AUTO_VIDEOS=true to run this long evaluation.',
      );
      return;
    }

    final repoRoot = Directory.current.parent.path;
    final videos = [
      p.join(repoRoot, 'test_videos', '20260503_000708.mp4'),
      p.join(repoRoot, 'test_videos', '20260503_000448.mp4'),
      p.join(repoRoot, 'test_videos', '20260503_000315.mp4'),
      p.join(repoRoot, 'test_videos', '20260503_000150.mp4'),
    ];
    final outDir = Directory(p.join(repoRoot, 'test_videos', 'trackt_auto_eval'));
    outDir.createSync(recursive: true);

    final summary = <Map<String, dynamic>>[];
    for (final video in videos) {
      final file = File(video);
      expect(file.existsSync(), isTrue, reason: video);

      final stem = p.basenameWithoutExtension(video);
      final outVideo = p.join(outDir.path, '$stem.auto.annotated.mp4');
      final result = await runPipeline(
        file.absolute.path,
        outVideo,
        options: const PipelineOptions(targetMode: TargetMode.auto),
      );

      expect(result.activeTargets, isNotEmpty,
          reason: '$stem: auto mode found no targets');

      for (final id in result.fallingOrder) {
        final ev = result.fallEvents[id];
        expect(ev, isNotNull, reason: '$stem missing fall_event for $id');
        expect(ev!.x, inInclusiveRange(0, result.originalWidth - 1));
        expect(ev.y, inInclusiveRange(0, result.originalHeight - 1));
      }

      final resultPath = p.join(outDir.path, '$stem.auto.result.json');
      File(resultPath).writeAsStringSync(
        const JsonEncoder.withIndent('  ').convert(result.toJson()),
      );
      summary.add({
        'input': file.absolute.path,
        'mode': 'auto',
        'discovered_targets': result.activeTargets,
        'discovered_count': result.activeTargets.length,
        'falling_order': result.fallingOrder,
        'score': result.score,
        'total_targets': result.totalTargets,
        'motion_path_points': result.carPathPoints,
        'fps': result.fps,
        'processing_time_ms': result.processingTime.inMilliseconds,
      });
    }

    File(p.join(outDir.path, 'summary.json')).writeAsStringSync(
      const JsonEncoder.withIndent('  ').convert(summary),
    );
  }, timeout: const Timeout(Duration(minutes: 45)));
}
