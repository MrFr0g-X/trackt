import 'package:flutter_test/flutter_test.dart';
import 'package:dsai352_cv/models/pipeline_result.dart';

void main() {
  test('PipelineOptions serializes defaults', () {
    const options = PipelineOptions();

    expect(options.confidenceThreshold, 0.7);
    expect(options.downscaleHeight, 360);
    expect(options.maxTargets, 6);
    expect(options.minDisplayConfidence, 0.55);
    expect(options.debugMode, false);
    expect(options.expectedTargetCount, 6);
    expect(options.targetMode, TargetMode.controlled);
    expect(options.illuminationNormalization, false);
    expect(options.dualStreamPreprocessing, true);
    expect(options.semanticAssist, false);
    expect(options.frameStep, 2);
    expect(options.writeAnnotatedVideo, true);
    expect(options.earlyStopOnComplete, false);
    expect(options.toJson(), {
      'confidence_threshold': 0.7,
      'downscale_height': 360,
      'max_targets': 6,
      'min_display_confidence': 0.55,
      'debug_mode': false,
      'expected_target_count': 6,
      'target_mode': 'controlled',
      'illumination_normalization': false,
      'dual_stream_preprocessing': true,
      'semantic_assist': false,
      'frame_step': 2,
      'write_annotated_video': true,
      'early_stop_on_complete': false,
    });
  });

  test('PipelineOptions mobileFast uses on-device speed defaults', () {
    const options = PipelineOptions.mobileFast();

    expect(options.downscaleHeight, 360);
    expect(options.dualStreamPreprocessing, true);
    expect(options.semanticAssist, false);
    expect(options.expectedTargetCount, isNull);
    expect(options.frameStep, 8);
    expect(options.earlyStopOnComplete, true);
  });
}
