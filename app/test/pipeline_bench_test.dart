import 'dart:io';

import 'package:dsai352_cv/models/pipeline_result.dart';
import 'package:dsai352_cv/pipeline/cv_pipeline.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

const _runBench = bool.fromEnvironment('TRACKT_BENCH');

void main() {
  const budgetSec = 90;

  test('controlled-mode pipeline finishes within performance budget', () async {
    if (!_runBench) {
      markTestSkipped(
        'Set --dart-define=TRACKT_BENCH=true to run the bench test.',
      );
      return;
    }

    final repoRoot = Directory.current.parent.path;
    final video = p.join(repoRoot, 'test_videos', '20260503_000150.mp4');
    expect(File(video).existsSync(), isTrue, reason: video);

    final outDir = Directory(p.join(repoRoot, 'test_videos', 'trackt_bench'));
    outDir.createSync(recursive: true);
    final outVideo = p.join(outDir.path, 'bench.annotated.mp4');

    final start = DateTime.now();
    final result = await runPipeline(
      video,
      outVideo,
      options: const PipelineOptions(),
    );
    final elapsed = DateTime.now().difference(start);

    expect(result.score, greaterThanOrEqualTo(1),
        reason: 'pipeline produced no fall events');
    expect(elapsed.inSeconds, lessThanOrEqualTo(budgetSec),
        reason:
            'pipeline took ${elapsed.inSeconds}s — over budget of ${budgetSec}s');
  }, timeout: const Timeout(Duration(minutes: 5)));
}
