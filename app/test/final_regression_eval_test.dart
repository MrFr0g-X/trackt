import 'dart:convert';
import 'dart:io';

import 'package:dsai352_cv/models/pipeline_result.dart';
import 'package:dsai352_cv/pipeline/cv_pipeline.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

const _runFinalEval = bool.fromEnvironment('TRACKT_EVAL_FINAL_SET');

void main() {
  test(
    'evaluate final car-only and car-near-target clips',
    () async {
      if (!_runFinalEval) {
        markTestSkipped(
          'Set --dart-define=TRACKT_EVAL_FINAL_SET=true to run this long evaluation.',
        );
        return;
      }

      final repoRoot = Directory.current.parent.path;
      final videos = [
        p.join(
          repoRoot,
          'test_videos',
          'Car-only videos',
          '20260505_203358.mp4',
        ),
        p.join(
          repoRoot,
          'test_videos',
          'Car-near-target videos',
          '20260505_202456.mp4',
        ),
        p.join(
          repoRoot,
          'test_videos',
          'Car-near-target videos',
          '20260505_202719.mp4',
        ),
        p.join(
          repoRoot,
          'test_videos',
          'Collision videos',
          '20260505_203825.mp4',
        ),
      ];

      final outDir = Directory(
        p.join(repoRoot, 'test_videos', 'trackt_final_eval'),
      );
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
        summary.add({
          'input': file.absolute.path,
          'result_json': resultPath,
          'score': result.score,
          'total_targets': result.totalTargets,
          'falling_order': result.fallingOrder,
          'path_drawn': result.pathDrawn,
          'motion_path_points': result.carPathPoints,
          'frames_processed': result.framesProcessed,
          'processing_time_ms': result.processingTime.inMilliseconds,
          'output_video': result.outputVideoPath,
        });
      }

      File(
        p.join(outDir.path, 'summary.json'),
      ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(summary));
    },
    timeout: const Timeout(Duration(minutes: 45)),
  );
}
