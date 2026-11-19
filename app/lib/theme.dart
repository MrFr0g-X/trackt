import 'package:flutter/material.dart';

const bgDark = Color(0xFF080807);
const bgMid = Color(0xFF12100D);
const panelBg = Color(0xFF1C1712);

const glass = Color(0x10FFFFFF);
const glassBorder = Color(0x1AFFFFFF);
const glassHover = Color(0x1AFFFFFF);

const accentPrimary = Color(0xFFFFB703);
const accentSecondary = Color(0xFF2DD4BF);
const accentWarm = Color(0xFFFF6B35);

const accentViolet = accentPrimary;
const accentPink = Color(0xFFEC4899);
const accentBlue = accentPrimary;
const accentCyan = accentSecondary;

const success = Color(0xFF22C55E);
const warning = Color(0xFFFBBF24);
const error = Color(0xFFF87171);

const textPrimary = Color(0xFFE2E8F0);
const textSecondary = Color(0xFF94A3B8);
const textMuted = Color(0xFF64748B);

const gradientPrimary = LinearGradient(
  colors: [Color(0xFFFFB703), Color(0xFFFF6B35)],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);

const gradientCool = LinearGradient(
  colors: [Color(0xFF1F2937), Color(0xFF0F766E)],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);

const spectrumColors = <Color>[
  Color(0xFFEF4444),
  Color(0xFFF59E0B),
  Color(0xFFFBBF24),
  Color(0xFF84CC16),
  Color(0xFF10B981),
  Color(0xFF14B8A6),
  Color(0xFF06B6D4),
  Color(0xFF3B82F6),
  Color(0xFF8B5CF6),
  Color(0xFFEC4899),
];

ThemeData buildTheme() {
  final base = ThemeData.dark();
  return base.copyWith(
    scaffoldBackgroundColor: bgDark,
    colorScheme: const ColorScheme.dark(
      primary: accentPrimary,
      secondary: accentSecondary,
      surface: bgMid,
      onSurface: textPrimary,
    ),
    textTheme: base.textTheme.apply(
      bodyColor: textPrimary,
      displayColor: textPrimary,
    ),
    iconTheme: const IconThemeData(color: textSecondary),
  );
}
