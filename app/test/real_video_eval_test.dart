import 'dart:convert';
import 'dart:io';

import 'package:dsai352_cv/models/pipeline_result.dart';
import 'package:dsai352_cv/pipeline/cv_pipeline.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

const _runRealVideoEval = bool.fromEnvironment('TRACKT_EVAL_REAL_VIDEOS');

void main() {
  test('evaluate real captured videos', () async {
    if (!_runRealVideoEval) {
      markTestSkipped(
        'Set --dart-define=TRACKT_EVAL_REAL_VIDEOS=true to run this long evaluation.',
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
    final outDir = Directory(p.join(repoRoot, 'test_videos', 'trackt_eval'));
    outDir.createSync(recursive: true);

    final summary = <Map<String, dynamic>>[];
    for (final video in videos) {
      final file = File(video);
      expect(file.existsSync(), isTrue, reason: video);

      final stem = p.basenameWithoutExtension(video);
      final outVideo = p.join(outDir.path, '$stem.annotated.mp4');
      final result = await runPipeline(
        file.absolute.path,
        outVideo,
        options: const PipelineOptions.mobileFast(),
      );

      final resultPath = p.join(outDir.path, '$stem.result.json');
      File(resultPath).writeAsStringSync(
        const JsonEncoder.withIndent('  ').convert(result.toJson()),
      );

      for (final id in result.fallingOrder) {
        final ev = result.fallEvents[id];
        expect(
          ev,
          isNotNull,
          reason: '$stem: missing fall_event for fallen target $id',
        );
        expect(
          ev!.x,
          inInclusiveRange(0, result.originalWidth - 1),
          reason: '$stem/$id x outside original frame',
        );
        expect(
          ev.y,
          inInclusiveRange(0, result.originalHeight - 1),
          reason: '$stem/$id y outside original frame',
        );
        expect(
          ev.signal,
          isIn(<String>['shape', 'area', 'motion', 'spread', 'vanish']),
          reason: '$stem/$id unknown fall signal',
        );
        expect(
          ev.frame,
          greaterThanOrEqualTo(0),
          reason: '$stem/$id non-negative frame',
        );
        expect(
          ev.tSec,
          greaterThanOrEqualTo(0.0),
          reason: '$stem/$id non-negative time',
        );
      }

      summary.add({
        'input': file.absolute.path,
        'result_json': resultPath,
        'score': result.score,
        'total_targets': result.totalTargets,
        'falling_order': result.fallingOrder,
        'fall_times_sec': result.fallTimesSec,
        'fall_events': result.fallEvents.map((k, v) => MapEntry(k, v.toJson())),
        'confidences': result.confidences,
        'path_drawn': result.pathDrawn,
        'motion_path_points': result.carPathPoints,
        'frames_processed': result.framesProcessed,
        'fps': result.fps,
        'processing_time_ms': result.processingTime.inMilliseconds,
        'output_video': result.outputVideoPath,
        'original_width': result.originalWidth,
        'original_height': result.originalHeight,
      });
    }

    File(
      p.join(outDir.path, 'summary.json'),
    ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(summary));
  }, timeout: const Timeout(Duration(minutes: 45)));
}
