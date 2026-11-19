import 'dart:io';

import 'package:opencv_dart/opencv_dart.dart' as cv;

void main(List<String> args) {
  if (args.length < 2) {
    stderr.writeln('Usage: dart run tool/extract_frames.dart <video> <out_dir>');
    exit(64);
  }

  final input = args[0];
  final outDir = Directory(args[1])..createSync(recursive: true);
  final cap = cv.VideoCapture.fromFile(input);
  if (!cap.isOpened) {
    stderr.writeln('Cannot open $input');
    exit(1);
  }

  final total = cap.get(cv.CAP_PROP_FRAME_COUNT).round();
  final frames = <int>{
    0,
    (total * 0.10).round(),
    (total * 0.35).round(),
    (total * 0.65).round(),
    (total * 0.90).round(),
  }.where((f) => f >= 0 && f < total).toList()
    ..sort();

  final stem = input
      .split(RegExp(r'[\\/]'))
      .last
      .replaceAll(RegExp(r'\.[^.]+$'), '');

  for (final frameNo in frames) {
    cap.set(cv.CAP_PROP_POS_FRAMES, frameNo.toDouble());
    final (ok, frame) = cap.read();
    if (!ok || frame.isEmpty) {
      frame.dispose();
      continue;
    }

    final path =
        '${outDir.path}${Platform.pathSeparator}${stem}_f${frameNo.toString().padLeft(5, '0')}.jpg';
    cv.imwrite(path, frame);
    frame.dispose();
    stdout.writeln(path);
  }

  cap.release();
}
