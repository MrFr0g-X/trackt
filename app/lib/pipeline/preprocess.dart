import 'package:opencv_dart/opencv_dart.dart' as cv;

class ScenePreprocessProfile {
  final bool enabled;
  final int sampledFrames;
  final double carLikePixelRatio;
  final double targetLikePixelRatio;
  final double avgCarHue;
  final double avgCarSaturation;
  final double avgCarValue;
  final int darkValueCutoff;

  const ScenePreprocessProfile({
    required this.enabled,
    required this.sampledFrames,
    required this.carLikePixelRatio,
    required this.targetLikePixelRatio,
    required this.avgCarHue,
    required this.avgCarSaturation,
    required this.avgCarValue,
    required this.darkValueCutoff,
  });

  const ScenePreprocessProfile.disabled()
    : enabled = false,
      sampledFrames = 0,
      carLikePixelRatio = 0,
      targetLikePixelRatio = 0,
      avgCarHue = 0,
      avgCarSaturation = 0,
      avgCarValue = 0,
      darkValueCutoff = 115;

  Map<String, dynamic> toJson() => {
    'enabled': enabled,
    'sampled_frames': sampledFrames,
    'car_like_pixel_ratio': carLikePixelRatio,
    'target_like_pixel_ratio': targetLikePixelRatio,
    'avg_car_hue': avgCarHue,
    'avg_car_saturation': avgCarSaturation,
    'avg_car_value': avgCarValue,
    'dark_value_cutoff': darkValueCutoff,
  };
}

ScenePreprocessProfile calibrateScenePreprocess(
  String inputPath, {
  required double scale,
  required int width,
  required int height,
  required Map<String, List<List<int>>> targetRanges,
  int sampleFrames = 24,
}) {
  final cap = cv.VideoCapture.fromFile(inputPath);
  if (!cap.isOpened) return const ScenePreprocessProfile.disabled();

  var scanned = 0;
  var totalPixels = 0;
  var carLike = 0;
  var targetLike = 0;
  var hSum = 0.0;
  var sSum = 0.0;
  var vSum = 0.0;

  for (int i = 0; i < sampleFrames; i++) {
    final (ok, raw) = cap.read();
    if (!ok || raw.isEmpty) {
      raw.dispose();
      break;
    }
    cv.Mat frame;
    if ((scale - 1.0).abs() > 0.001) {
      frame = cv.resize(raw, (width, height), interpolation: cv.INTER_AREA);
      raw.dispose();
    } else {
      frame = raw;
    }

    final hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV);
    frame.dispose();
    final h = cv.extractChannel(hsv, 0);
    final s = cv.extractChannel(hsv, 1);
    final v = cv.extractChannel(hsv, 2);
    hsv.dispose();

    final hd = h.data;
    final sd = s.data;
    final vd = v.data;
    totalPixels += hd.length;
    for (int p = 0; p < hd.length; p++) {
      final hh = hd[p], ss = sd[p], vv = vd[p];
      final isTarget = _pixelInTargetRanges(hh, ss, vv, targetRanges);
      if (isTarget) {
        targetLike++;
        continue;
      }
      final dark = vv < 120 && ss > 10;
      final blueDark = hh >= 86 && hh <= 134 && ss > 35 && vv < 225;
      if (dark || blueDark) {
        carLike++;
        hSum += hh;
        sSum += ss;
        vSum += vv;
      }
    }
    h.dispose();
    s.dispose();
    v.dispose();
    scanned++;
  }
  cap.release();

  if (scanned == 0 || totalPixels == 0 || carLike == 0) {
    return const ScenePreprocessProfile.disabled();
  }

  final avgV = vSum / carLike;
  final cutoff = avgV < 65
      ? 135
      : avgV < 90
      ? 125
      : 115;

  return ScenePreprocessProfile(
    enabled: true,
    sampledFrames: scanned,
    carLikePixelRatio: carLike / totalPixels,
    targetLikePixelRatio: targetLike / totalPixels,
    avgCarHue: hSum / carLike,
    avgCarSaturation: sSum / carLike,
    avgCarValue: avgV,
    darkValueCutoff: cutoff,
  );
}

cv.Mat enhanceTargetsForModel(
  cv.Mat bgr,
  Map<String, List<List<int>>> targetRanges,
) {
  cv.Mat? hsv, combinedMask;
  final out = bgr.clone();
  try {
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV);
    for (final entry in targetRanges.entries) {
      final mask = _maskForRanges(hsv, entry.value);
      if (combinedMask == null) {
        combinedMask = mask.clone();
      } else {
        final merged = cv.bitwiseOR(combinedMask, mask);
        combinedMask.dispose();
        combinedMask = merged;
      }

      final color = cv.Mat.fromScalar(
        bgr.rows,
        bgr.cols,
        bgr.type,
        _canonicalBgr(entry.key),
      );
      color.copyTo(out, mask: mask);
      color.dispose();
      mask.dispose();
    }
  } catch (_) {
    return out;
  } finally {
    hsv?.dispose();
    combinedMask?.dispose();
  }
  return out;
}

cv.Mat enhanceCarForModel(
  cv.Mat bgr,
  Map<String, List<List<int>>> targetRanges,
  ScenePreprocessProfile profile,
) {
  cv.Mat? norm;
  cv.Mat? dim;
  cv.Mat? boosted;
  cv.Mat? hsv;
  cv.Mat? h;
  cv.Mat? s;
  cv.Mat? v;
  cv.Mat? darkMask;
  cv.Mat? satMask;
  cv.Mat? blueHue;
  cv.Mat? blueSat;
  cv.Mat? blueVal;
  cv.Mat? blueMask;
  cv.Mat? carMask;
  cv.Mat? targetMask;
  cv.Mat? notTarget;
  cv.Mat? carOnly;
  cv.Mat? closed;
  try {
    norm = applyIlluminationNorm(bgr, clipLimit: 2.4);
    dim = cv.convertScaleAbs(norm, alpha: 0.72, beta: -8);
    boosted = cv.convertScaleAbs(norm, alpha: 1.85, beta: 42);

    hsv = cv.cvtColor(norm, cv.COLOR_BGR2HSV);
    h = cv.extractChannel(hsv, 0);
    s = cv.extractChannel(hsv, 1);
    v = cv.extractChannel(hsv, 2);

    final darkCutoff = profile.darkValueCutoff.clamp(95, 145).toInt();
    darkMask = _ch1Range(v, 0, darkCutoff);
    satMask = _ch1Range(s, 10, 255);
    final darkSat = cv.bitwiseAND(darkMask, satMask);

    blueHue = _ch1Range(h, 86, 134);
    blueSat = _ch1Range(s, 35, 255);
    blueVal = _ch1Range(v, 15, 235);
    final blueSv = cv.bitwiseAND(blueSat, blueVal);
    blueMask = cv.bitwiseAND(blueHue, blueSv);
    blueSv.dispose();

    carMask = cv.bitwiseOR(darkSat, blueMask);
    darkSat.dispose();

    targetMask = _maskForRanges(hsv, [
      for (final ranges in targetRanges.values) ...ranges,
    ]);
    notTarget = cv.bitwiseNOT(targetMask);
    carOnly = cv.bitwiseAND(carMask, notTarget);
    final kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5));
    closed = cv.morphologyEx(carOnly, cv.MORPH_CLOSE, kernel);
    kernel.dispose();

    boosted.copyTo(dim, mask: closed);
    return dim;
  } catch (_) {
    dim?.dispose();
    return bgr.clone();
  } finally {
    norm?.dispose();
    boosted?.dispose();
    hsv?.dispose();
    h?.dispose();
    s?.dispose();
    v?.dispose();
    darkMask?.dispose();
    satMask?.dispose();
    blueHue?.dispose();
    blueSat?.dispose();
    blueVal?.dispose();
    blueMask?.dispose();
    carMask?.dispose();
    targetMask?.dispose();
    notTarget?.dispose();
    carOnly?.dispose();
    closed?.dispose();
  }
}

bool _pixelInTargetRanges(
  int h,
  int s,
  int v,
  Map<String, List<List<int>>> targetRanges,
) {
  for (final ranges in targetRanges.values) {
    for (final r in ranges) {
      if (h >= r[0] && h <= r[1] && s >= r[2] && v >= r[3]) return true;
    }
  }
  return false;
}

cv.Mat _maskForRanges(cv.Mat hsv, List<List<int>> ranges) {
  final h = cv.extractChannel(hsv, 0);
  final s = cv.extractChannel(hsv, 1);
  final v = cv.extractChannel(hsv, 2);
  cv.Mat? combined;
  try {
    for (final r in ranges) {
      final hM = _ch1Range(h, r[0], r[1]);
      final sM = _ch1Range(s, r[2], 255);
      final vM = _ch1Range(v, r[3], 255);
      final sv = cv.bitwiseAND(sM, vM);
      final part = cv.bitwiseAND(hM, sv);
      if (combined == null) {
        combined = part.clone();
      } else {
        final merged = cv.bitwiseOR(combined, part);
        combined.dispose();
        combined = merged;
      }
      hM.dispose();
      sM.dispose();
      vM.dispose();
      sv.dispose();
      part.dispose();
    }
    return combined ?? cv.Mat.zeros(hsv.rows, hsv.cols, cv.MatType.CV_8UC1);
  } finally {
    h.dispose();
    s.dispose();
    v.dispose();
  }
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
  final out = cv.inRange(ch, loM, hiM);
  loM.dispose();
  hiM.dispose();
  return out;
}

cv.Scalar _canonicalBgr(String key) {
  final k = key.toLowerCase();
  if (k.contains('red')) return cv.Scalar(0, 0, 255);
  if (k.contains('blue') || k.contains('cyan')) return cv.Scalar(255, 0, 0);
  if (k.contains('yellow')) return cv.Scalar(0, 255, 255);
  if (k.contains('green')) return cv.Scalar(0, 255, 0);
  if (k.contains('orange')) return cv.Scalar(0, 128, 255);
  if (k.contains('purple') || k.contains('magenta')) {
    return cv.Scalar(255, 0, 255);
  }
  return cv.Scalar(255, 255, 255);
}

cv.Mat applyIlluminationNorm(
  cv.Mat bgr, {
  double clipLimit = 2.0,
  (int, int) tileGridSize = (8, 8),
}) {
  cv.Mat? lab, l, a, b, lEq, merged;
  cv.Mat? out;
  try {
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab);
    l = cv.extractChannel(lab, 0);
    a = cv.extractChannel(lab, 1);
    b = cv.extractChannel(lab, 2);

    final claheOp = cv.createCLAHE(
      clipLimit: clipLimit,
      tileGridSize: tileGridSize,
    );
    lEq = claheOp.apply(l);
    claheOp.dispose();

    merged = cv.merge(cv.VecMat.fromList([lEq, a, b]));
    out = cv.cvtColor(merged, cv.COLOR_Lab2BGR);
  } catch (_) {
    out = bgr.clone();
  } finally {
    lab?.dispose();
    l?.dispose();
    a?.dispose();
    b?.dispose();
    lEq?.dispose();
    merged?.dispose();
  }
  return out;
}
