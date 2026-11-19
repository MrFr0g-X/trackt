import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:permission_handler/permission_handler.dart';

import '../theme.dart';
import '../models/pipeline_result.dart';
import '../pipeline/cv_pipeline.dart';
import '../pipeline/live_yolo_detector.dart';
import '../widgets/glass_card.dart';
import '../widgets/dashboard_widgets.dart';
import 'live_camera_screen.dart';

enum _AppState { idle, selected, processing, done, error }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  _AppState _state = _AppState.idle;
  String? _inputPath;
  String? _jsonOutputPath;
  String? _errorMessage;
  PipelineResult? _result;
  PipelineOptions get _options => const PipelineOptions.v2Hybrid();

  late final Player _player;
  late final VideoController _videoCtrl;
  bool _videoReady = false;
  bool _showingOutput = false;
  Duration _videoPosition = Duration.zero;
  Duration _videoDuration = Duration.zero;
  StreamSubscription? _posSub;
  StreamSubscription? _durSub;

  PipelineProgress? _progress;
  final _liveLog = <LogEvent>[];
  final _liveCounts = <CountSample>[];
  final _liveConfidences = <String, double>{};

  late final AnimationController _spinCtrl;

  @override
  void initState() {
    super.initState();
    _spinCtrl = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();

    _player = Player();
    _videoCtrl = VideoController(_player);
    _posSub = _player.stream.position.listen((d) {
      if (mounted) setState(() => _videoPosition = d);
    });
    _durSub = _player.stream.duration.listen((d) {
      if (mounted) setState(() => _videoDuration = d);
    });

    _requestPermissions();
    if (Platform.isAndroid || Platform.isIOS) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        unawaited(LiveYoloDetector.load());
      });
    }
  }

  @override
  void dispose() {
    _spinCtrl.dispose();
    _posSub?.cancel();
    _durSub?.cancel();
    _player.dispose();
    super.dispose();
  }

  Future<void> _requestPermissions() async {
    if (Platform.isAndroid) {
      await [
        Permission.storage,
        Permission.videos,
        Permission.photos,
        Permission.camera,
      ].request();
    }
  }

  Future<void> _pickVideo() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );
    if (res == null || res.files.isEmpty) return;
    final path = res.files.single.path;
    if (path == null) return;

    await _selectVideoPath(path);
  }

  Future<void> _recordVideo() async {
    if (!Platform.isAndroid && !Platform.isIOS) return;
    final video = await ImagePicker().pickVideo(
      source: ImageSource.camera,
      maxDuration: const Duration(minutes: 3),
    );
    if (video == null) return;
    await _selectVideoPath(video.path);
  }

  Future<void> _openLiveCamera() async {
    if (!Platform.isAndroid && !Platform.isIOS) return;
    if (_state == _AppState.processing) return;
    await Permission.camera.request();
    if (!mounted) return;
    await Navigator.of(
      context,
    ).push(MaterialPageRoute(builder: (_) => const LiveCameraScreen()));
  }

  Future<void> _selectVideoPath(String path) async {
    await _loadVideoPreview(path);
    setState(() {
      _inputPath = path;
      _jsonOutputPath = null;
      _state = _AppState.selected;
      _result = null;
      _showingOutput = false;
      _errorMessage = null;
      _liveLog.clear();
      _liveCounts.clear();
      _liveConfidences.clear();
      _progress = null;
    });
  }

  Future<void> _loadVideoPreview(String path) async {
    setState(() => _videoReady = false);
    try {
      await _player.open(Media('file:///${path.replaceAll('\\', '/')}'));
      await _player.setPlaylistMode(PlaylistMode.loop);
      setState(() => _videoReady = true);
    } catch (_) {
    }
  }

  Future<void> _runPipeline() async {
    if (_inputPath == null) return;
    setState(() {
      _state = _AppState.processing;
      _showingOutput = false;
      _liveLog.clear();
      _liveCounts.clear();
      _liveConfidences.clear();
      _progress = const PipelineProgress(PipelineStage.load, 0, 'Starting...');
    });

    final outDir = await _outputDirectory();
    final outPath = p.join(outDir, 'dsai352_annotated.mp4');
    final partialPath = p.join(outDir, 'dsai352_result.partial.json');

    DateTime lastPartialWrite = DateTime.fromMillisecondsSinceEpoch(0);
    Future<void> writePartial() async {
      final now = DateTime.now();
      if (now.difference(lastPartialWrite).inMilliseconds < 5000) return;
      lastPartialWrite = now;
      try {
        final partial = {
          'partial': true,
          'state': _state.name,
          'input': _inputPath,
          'progress_stage': _progress?.stage.name,
          'progress_pct': _progress?.progress,
          'progress_message': _progress?.message,
          'event_log': _liveLog.map((e) => e.toJson()).toList(),
          'count_timeline': _liveCounts.map((c) => c.toJson()).toList(),
          'live_confidences': _liveConfidences,
          'updated_at': now.toIso8601String(),
        };
        await File(
          partialPath,
        ).writeAsString(const JsonEncoder.withIndent('  ').convert(partial));
      } catch (_) {
      }
    }

    try {
      final result = await runPipeline(
        _inputPath!,
        outPath,
        options: _options,
        onProgress: (pr) {
          if (!mounted) return;
          setState(() => _progress = pr);
          writePartial();
        },
        onLog: (ev) {
          if (!mounted) return;
          setState(() {
            _liveLog.add(ev);
            if (ev.color != null) {
              _liveConfidences[ev.color!] = (_liveConfidences[ev.color!] ?? 0.5)
                  .clamp(0.0, 1.0);
            }
          });
          writePartial();
        },
        onCount: (cs) {
          if (!mounted) return;
          setState(() => _liveCounts.add(cs));
          writePartial();
        },
      );
      if (!mounted) return;
      setState(() {
        _result = result;
        _state = _AppState.done;
        _liveConfidences
          ..clear()
          ..addAll(result.confidences);
      });
      final jsonPath = p.join(outDir, 'dsai352_result.json');
      await File(jsonPath).writeAsString(
        const JsonEncoder.withIndent('  ').convert(result.toJson()),
      );
      try {
        final partial = File(partialPath);
        if (await partial.exists()) await partial.delete();
      } catch (_) {
      }
      final hasAnnotatedOutput =
          result.outputVideoPath.isNotEmpty &&
          await File(result.outputVideoPath).exists();
      await _loadVideoPreview(
        hasAnnotatedOutput ? result.outputVideoPath : _inputPath!,
      );
      setState(() {
        _jsonOutputPath = jsonPath;
        _showingOutput = hasAnnotatedOutput;
      });
    } catch (e, st) {
      if (!mounted) return;
      String? errorPath;
      try {
        final dir = await _outputDirectory();
        errorPath = p.join(dir, 'dsai352_error.txt');
        await File(errorPath).writeAsString(
          'Input: $_inputPath\n'
          'Time: ${DateTime.now().toIso8601String()}\n\n'
          '$e\n\n$st',
        );
      } catch (_) {
      }
      setState(() {
        _state = _AppState.error;
        _errorMessage = '$e\n\n$st';
        _jsonOutputPath = errorPath;
      });
    }
  }

  Future<String> _outputDirectory() async {
    if (Platform.isAndroid) {
      final dir = await getExternalStorageDirectory();
      if (dir != null) {
        final out = Directory(p.join(dir.path, 'outputs'));
        await out.create(recursive: true);
        return out.path;
      }
    }
    final dir = await getTemporaryDirectory();
    return dir.path;
  }

  void _togglePlayPause() =>
      _player.state.playing ? _player.pause() : _player.play();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          final compact =
              Platform.isAndroid ||
              constraints.maxHeight < 560 ||
              constraints.maxWidth < 1120;
          return Stack(
            children: [
              _buildBackground(),
              SafeArea(
                child: Padding(
                  padding: EdgeInsets.all(compact ? 8 : 12),
                  child: compact
                      ? _buildPhoneLayout(constraints)
                      : _buildDesktopLayout(),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildDesktopLayout() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(width: 300, child: _buildLeftPanel()),
        const SizedBox(width: 12),
        Expanded(child: _buildCenterPanel()),
        const SizedBox(width: 12),
        SizedBox(width: 340, child: _buildRightPanel()),
      ],
    );
  }

  Widget _buildPhoneLayout(BoxConstraints constraints) {
    final videoHeight = (constraints.maxWidth * 0.62).clamp(220.0, 330.0);
    return ListView(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.only(bottom: 18),
      children: [
        _buildPhoneHeader(),
        const SizedBox(height: 14),
        _buildMobileActionDeck(),
        const SizedBox(height: 14),
        SizedBox(height: videoHeight, child: _buildVideoCard(compact: true)),
        const SizedBox(height: 14),
        _buildMobileOutcomeDeck(),
        const SizedBox(height: 14),
        _buildMobilePlaybackStrip(),
      ],
    );
  }

  Widget _buildPhoneHeader() {
    final subtitle = switch (_state) {
      _AppState.idle => 'Select or record a clip',
      _AppState.selected => 'Ready for scoring',
      _AppState.processing => 'Processing the scene',
      _AppState.done => 'Result ready',
      _AppState.error => 'Review the error log',
    };
    return Container(
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: const LinearGradient(
          colors: [Color(0xFF2A2113), Color(0xFF15120D)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              color: Colors.black.withValues(alpha: 0.22),
              border: Border.all(color: accentPrimary.withValues(alpha: 0.35)),
            ),
            child: const Icon(Icons.center_focus_strong, color: accentPrimary),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Vision Scorer',
                  style: TextStyle(
                    color: textPrimary,
                    fontSize: 22,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  subtitle,
                  style: const TextStyle(color: textSecondary, fontSize: 12),
                ),
              ],
            ),
          ),
          if (_result != null)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: accentPrimary.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '${_result!.score}/${_result!.totalTargets}',
                style: const TextStyle(
                  color: accentPrimary,
                  fontSize: 18,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMobileActionDeck() {
    final canUseVideo = _state != _AppState.processing;
    return GlassCard(
      borderRadius: 28,
      padding: const EdgeInsets.all(16),
      glassColor: panelBg.withValues(alpha: 0.92),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Text(
                'Clip source',
                style: TextStyle(
                  color: textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w900,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 5,
                ),
                decoration: BoxDecoration(
                  color: accentSecondary.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: const Text(
                  'smart default',
                  style: TextStyle(
                    color: accentSecondary,
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_inputPath == null)
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.20),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
              ),
              child: const Row(
                children: [
                  Icon(Icons.movie_creation_outlined, color: textMuted),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'No video selected',
                      style: TextStyle(
                        color: textSecondary,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            )
          else
            _SelectedClipCard(
              name: p.basename(_inputPath!),
              state: _state,
              progress: _progress?.progress ?? 0,
              stage: _progress?.stage,
            ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _CommandTile(
                  label: 'Gallery',
                  icon: Icons.folder_rounded,
                  enabled: canUseVideo,
                  onTap: _pickVideo,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _CommandTile(
                  label: 'Camera',
                  icon: Icons.videocam_rounded,
                  enabled:
                      canUseVideo && (Platform.isAndroid || Platform.isIOS),
                  onTap: _recordVideo,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _CommandTile(
                  label: 'Live',
                  icon: Icons.sensors_rounded,
                  enabled:
                      canUseVideo && (Platform.isAndroid || Platform.isIOS),
                  onTap: _openLiveCamera,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          _buildRunButton(),
          if (_jsonOutputPath != null) ...[
            const SizedBox(height: 9),
            Text(
              p.basename(_jsonOutputPath!),
              textAlign: TextAlign.center,
              style: const TextStyle(color: textMuted, fontSize: 10),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    ).animate().fadeIn(duration: 350.ms).slideY(begin: 0.04, end: 0);
  }

  Widget _buildMobileOutcomeDeck() {
    final r = _result;
    final pct = r != null && r.totalTargets > 0
        ? (r.score / r.totalTargets).clamp(0.0, 1.0).toDouble()
        : 0.0;
    final progress = _progress;
    return GlassCard(
      borderRadius: 28,
      padding: const EdgeInsets.all(16),
      glassColor: panelBg.withValues(alpha: 0.92),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Text(
                'Outcome',
                style: TextStyle(
                  color: textPrimary,
                  fontSize: 16,
                  fontWeight: FontWeight.w900,
                ),
              ),
              const Spacer(),
              Text(
                _state == _AppState.processing
                    ? 'running'
                    : (r == null ? 'waiting' : 'complete'),
                style: const TextStyle(
                  color: textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          if (_state == _AppState.processing && progress != null) ...[
            _ProcessingStatus(progress: progress),
            const SizedBox(height: 12),
          ],
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.20),
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  r == null
                      ? 'Ready when you are'
                      : 'Score ${r.score}/${r.totalTargets}',
                  style: const TextStyle(
                    color: textPrimary,
                    fontSize: 24,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(999),
                  child: LinearProgressIndicator(
                    minHeight: 9,
                    value: pct,
                    backgroundColor: Colors.white.withValues(alpha: 0.07),
                    valueColor: const AlwaysStoppedAnimation(accentPrimary),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  r == null
                      ? 'Choose a clip, then start the analyzer.'
                      : r.fallingOrder.isEmpty
                      ? 'No falling order recorded.'
                      : 'Order: ${r.fallingOrderString}',
                  style: const TextStyle(color: textSecondary, fontSize: 12),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _ResultBadge(
                icon: Icons.flag_rounded,
                label: 'targets',
                value: r == null ? '—' : '${r.totalTargets}',
              ),
              _ResultBadge(
                icon: Icons.route_rounded,
                label: 'path',
                value: r == null ? '—' : (r.pathDrawn ? 'drawn' : 'missing'),
              ),
              _ResultBadge(
                icon: Icons.timer_rounded,
                label: 'time',
                value: r?.processingTimeString ?? '—',
              ),
            ],
          ),
          if (_state == _AppState.error && _errorMessage != null) ...[
            const SizedBox(height: 10),
            Text(
              _errorMessage!.split('\n').first,
              style: const TextStyle(color: error, fontSize: 11),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    ).animate().fadeIn(duration: 350.ms).slideY(begin: 0.04, end: 0);
  }

  Widget _buildMobilePlaybackStrip() {
    final totalMs = _videoDuration.inMilliseconds;
    final posMs = _videoPosition.inMilliseconds.clamp(0, totalMs);
    final value = totalMs > 0 ? posMs / totalMs : 0.0;
    return GlassCard(
      borderRadius: 24,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      glassColor: panelBg.withValues(alpha: 0.82),
      child: Row(
        children: [
          SizedBox(
            width: 44,
            height: 44,
            child: IconButton(
              style: IconButton.styleFrom(
                backgroundColor: accentPrimary.withValues(alpha: 0.12),
                foregroundColor: accentPrimary,
              ),
              onPressed: _videoReady ? _togglePlayPause : null,
              icon: StreamBuilder<bool>(
                stream: _player.stream.playing,
                initialData: _player.state.playing,
                builder: (_, s) => Icon(
                  (s.data ?? false)
                      ? Icons.pause_rounded
                      : Icons.play_arrow_rounded,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(999),
                  child: LinearProgressIndicator(
                    minHeight: 6,
                    value: value,
                    backgroundColor: Colors.white.withValues(alpha: 0.08),
                    valueColor: const AlwaysStoppedAnimation(accentSecondary),
                  ),
                ),
                const SizedBox(height: 7),
                Row(
                  children: [
                    Text(
                      _formatClock(_videoPosition),
                      style: const TextStyle(color: textMuted, fontSize: 10),
                    ),
                    const Spacer(),
                    Text(
                      _formatClock(_videoDuration),
                      style: const TextStyle(color: textMuted, fontSize: 10),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatClock(Duration d) {
    final mm = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final ss = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$mm:$ss';
  }

  Widget _buildBackground() {
    return Stack(
      children: [
        Positioned.fill(
          child: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFF080807), Color(0xFF1A140C)],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLeftPanel() {
    return GlassCard(
      padding: const EdgeInsets.all(14),
      child: _buildInputContent(),
    ).animate().fadeIn(duration: 400.ms).slideX(begin: -0.1, end: 0);
  }

  Widget _buildInputContent({bool compact = false}) {
    final content = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: compact ? MainAxisSize.min : MainAxisSize.max,
      children: [
        if (!compact) const SectionHeader(title: 'Input'),

        InkWell(
          onTap: _state == _AppState.processing ? null : _pickVideo,
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding: EdgeInsets.symmetric(
              vertical: compact ? 14 : 24,
              horizontal: 12,
            ),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: _inputPath != null
                    ? accentPrimary.withValues(alpha: 0.65)
                    : Colors.white.withValues(alpha: 0.12),
                width: 1.5,
              ),
              color: _inputPath != null
                  ? accentPrimary.withValues(alpha: 0.08)
                  : Colors.white.withValues(alpha: 0.02),
            ),
            child: Column(
              children: [
                Icon(
                  _inputPath == null
                      ? Icons.upload_file_rounded
                      : Icons.video_file_rounded,
                  size: compact ? 26 : 34,
                  color: _inputPath != null ? accentPrimary : textMuted,
                ),
                const SizedBox(height: 8),
                Text(
                  _inputPath == null ? 'Select video' : 'Change video',
                  style: TextStyle(
                    color: textSecondary,
                    fontSize: compact ? 12 : 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: _SmallActionButton(
                label: 'File',
                icon: Icons.folder_open_rounded,
                enabled: _state != _AppState.processing,
                onTap: _pickVideo,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _SmallActionButton(
                label: 'Record',
                icon: Icons.videocam_rounded,
                enabled:
                    _state != _AppState.processing &&
                    (Platform.isAndroid || Platform.isIOS),
                onTap: _recordVideo,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        _SmallActionButton(
          label: 'Open live view',
          icon: Icons.linked_camera_rounded,
          enabled:
              _state != _AppState.processing &&
              (Platform.isAndroid || Platform.isIOS),
          onTap: _openLiveCamera,
        ),

        const SizedBox(height: 14),
        const Text(
          'Selected file',
          style: TextStyle(
            color: textMuted,
            fontSize: 10,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.0,
          ),
        ),
        const SizedBox(height: 8),
        if (_inputPath != null)
          _InputFileTile(
            name: p.basename(_inputPath!),
            state: _state,
            progress: _progress?.progress ?? 0,
            stage: _progress?.stage,
          )
        else
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.02),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Text(
              'No input selected',
              style: TextStyle(color: textMuted, fontSize: 11),
            ),
          ),
        if (_jsonOutputPath != null) ...[
          const SizedBox(height: 8),
          Text(
            '${_state == _AppState.error ? "Error log" : "Result JSON"}: ${p.basename(_jsonOutputPath!)}',
            style: const TextStyle(color: textMuted, fontSize: 10),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],

        if (!compact) const Spacer() else const SizedBox(height: 14),

        if (compact)
          SizedBox(height: 46, child: _buildRunButton())
        else
          _buildRunButton(),
      ],
    );

    if (!compact) return content;
    return ListView(padding: EdgeInsets.zero, children: [content]);
  }

  Widget _buildRunButton() {
    return GradientButton(
      label: _state == _AppState.processing ? 'Analyzing...' : 'Analyze clip',
      icon: _state == _AppState.processing
          ? Icons.hourglass_bottom_rounded
          : Icons.bolt_rounded,
      enabled: _inputPath != null && _state != _AppState.processing,
      onPressed: _runPipeline,
    );
  }

  Widget _buildCenterPanel() {
    return Column(
      children: [
        Expanded(flex: 5, child: _buildVideoCard()),
        const SizedBox(height: 12),
        Expanded(flex: 3, child: _buildTimelineCard()),
      ],
    );
  }

  Widget _buildVideoCard({bool compact = false}) {
    return GlassCard(
      padding: EdgeInsets.all(compact ? 10 : 12),
      borderRadius: compact ? 28 : 16,
      glassColor: compact ? panelBg.withValues(alpha: 0.90) : glass,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Text(
                compact ? 'Preview' : 'Video workspace',
                style: TextStyle(
                  color: textPrimary,
                  fontSize: compact ? 16 : 14,
                  fontWeight: FontWeight.w900,
                ),
              ),
              const Spacer(),
              if (!compact)
                const Text(
                  'Video Player',
                  style: TextStyle(color: textMuted, fontSize: 11),
                ),
            ],
          ),
          SizedBox(height: compact ? 6 : 8),
          Expanded(child: _buildVideoPanel(compact: compact)),
        ],
      ),
    ).animate().fadeIn(duration: 500.ms);
  }

  Widget _buildTimelineCard({bool compact = false}) {
    return GlassCard(
      padding: EdgeInsets.all(compact ? 10 : 14),
      child: _buildTimelineEvents(compact: compact),
    ).animate(delay: 100.ms).fadeIn(duration: 500.ms);
  }

  Widget _buildVideoPanel({bool compact = false}) {
    if (!_videoReady) {
      return Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          color: Colors.black.withValues(alpha: 0.3),
        ),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.smart_display_outlined,
                size: compact ? 34 : 48,
                color: Colors.white.withValues(alpha: 0.15),
              ),
              SizedBox(height: compact ? 8 : 12),
              Text(
                _state == _AppState.processing
                    ? 'Processing video...'
                    : 'No video selected',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.25),
                  fontSize: compact ? 12 : 14,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        children: [
          Positioned.fill(
            child: Container(
              color: Colors.black,
              child: Video(
                controller: _videoCtrl,
                controls: NoVideoControls,
                fit: BoxFit.contain,
              ),
            ),
          ),
          Positioned(
            top: 8,
            left: 8,
            child: _showingOutput
                ? const StatusBadge(label: 'RESULT', color: success)
                : const StatusBadge(label: 'SOURCE', color: accentPrimary),
          ),
          Positioned(
            bottom: 8,
            right: 8,
            child: GestureDetector(
              onTap: _togglePlayPause,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(30),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: 0.15),
                  ),
                ),
                child: StreamBuilder<bool>(
                  stream: _player.stream.playing,
                  initialData: _player.state.playing,
                  builder: (_, snap) => Icon(
                    (snap.data ?? false)
                        ? Icons.pause_rounded
                        : Icons.play_arrow_rounded,
                    color: Colors.white,
                    size: 22,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTimelineEvents({bool compact = false}) {
    final progress = _progress;
    if (compact) {
      return SingleChildScrollView(
        padding: EdgeInsets.zero,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Text(
                  'Timeline',
                  style: TextStyle(
                    color: textPrimary,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                if (_state == _AppState.processing && progress != null)
                  Text(
                    '${(progress.progress * 100).round()}%',
                    style: const TextStyle(
                      color: textSecondary,
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 2),
            Row(
              children: [
                SizedBox(
                  width: 36,
                  child: IconButton(
                    padding: EdgeInsets.zero,
                    icon: StreamBuilder<bool>(
                      stream: _player.stream.playing,
                      initialData: _player.state.playing,
                      builder: (_, s) => Icon(
                        (s.data ?? false)
                            ? Icons.pause_rounded
                            : Icons.play_arrow_rounded,
                        color: textPrimary,
                        size: 20,
                      ),
                    ),
                    onPressed: _videoReady ? _togglePlayPause : null,
                  ),
                ),
                Expanded(
                  child: EventTimeline(
                    fallTimes: const {},
                    totalDuration: _videoDuration,
                    position: _videoPosition,
                    onSeek: (d) => _player.seek(d),
                    targetColors: const {},
                  ),
                ),
              ],
            ),
            if (_state == _AppState.processing && progress != null) ...[
              const SizedBox(height: 2),
              Text(
                progress.message,
                style: const TextStyle(color: textMuted, fontSize: 10),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 5),
              ClipRRect(
                borderRadius: BorderRadius.circular(3),
                child: LinearProgressIndicator(
                  minHeight: 5,
                  value: progress.progress.clamp(0.0, 1.0),
                  backgroundColor: Colors.white.withValues(alpha: 0.06),
                  valueColor: const AlwaysStoppedAnimation(accentPrimary),
                ),
              ),
            ],
          ],
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Row(
          children: [
            Text(
              'Timeline & Events',
              style: TextStyle(
                color: textPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w700,
              ),
            ),
            Spacer(),
            Text('Playback', style: TextStyle(color: textMuted, fontSize: 11)),
          ],
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            IconButton(
              icon: StreamBuilder<bool>(
                stream: _player.stream.playing,
                initialData: _player.state.playing,
                builder: (_, s) => Icon(
                  (s.data ?? false)
                      ? Icons.pause_rounded
                      : Icons.play_arrow_rounded,
                  color: textPrimary,
                  size: 20,
                ),
              ),
              onPressed: _videoReady ? _togglePlayPause : null,
            ),
            Expanded(
              child: EventTimeline(
                fallTimes: _result?.fallTimesSec ?? const {},
                totalDuration: _videoDuration,
                position: _videoPosition,
                onSeek: (d) => _player.seek(d),
                targetColors: _result?.targetColors ?? const {},
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),

        if (_state == _AppState.processing && progress != null) ...[
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.03),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        progress.message,
                        style: const TextStyle(
                          color: textSecondary,
                          fontSize: 11,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Text(
                      '${(progress.progress * 100).round()}%',
                      style: const TextStyle(
                        color: textPrimary,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ClipRRect(
                  borderRadius: BorderRadius.circular(3),
                  child: LinearProgressIndicator(
                    minHeight: 5,
                    value: progress.progress.clamp(0.0, 1.0),
                    backgroundColor: Colors.white.withValues(alpha: 0.06),
                    valueColor: const AlwaysStoppedAnimation(accentPrimary),
                  ),
                ),
              ],
            ),
          ),
        ],

        const SizedBox(height: 14),

        Expanded(
          child: Row(
            children: [
              Expanded(
                child: _SummaryTile(
                  label: 'Falling Order',
                  child: ColorChipRow(
                    sequence: _result?.fallingOrder ?? const [],
                    targetColors: _result?.targetColors ?? const {},
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _SummaryTile(
                  label: 'Processing Time',
                  child: Row(
                    children: [
                      const Icon(
                        Icons.timer_outlined,
                        color: accentSecondary,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _result?.processingTimeString ??
                            (_state == _AppState.processing ? '...' : '—'),
                        style: const TextStyle(
                          color: textPrimary,
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _SummaryTile(
                  label: 'Frames',
                  child: Row(
                    children: [
                      const Icon(
                        Icons.movie_filter_outlined,
                        color: accentPrimary,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _result != null
                            ? '${_result!.framesProcessed}'
                            : (_state == _AppState.processing ? '...' : '—'),
                        style: const TextStyle(
                          color: textPrimary,
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _SummaryTile(
                  label: 'Motion Path',
                  child: Row(
                    children: [
                      const Icon(
                        Icons.timeline_rounded,
                        color: accentWarm,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _result != null ? '${_result!.carPathPoints}' : '—',
                        style: const TextStyle(
                          color: textPrimary,
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildRightPanel() {
    return GlassCard(
      padding: const EdgeInsets.all(14),
      child: _buildAnalysisContent(),
    ).animate().fadeIn(duration: 400.ms).slideX(begin: 0.1, end: 0);
  }

  Widget _buildAnalysisContent({bool compact = false}) {
    final trackedTargets =
        (_result?.activeTargets ?? const <String>[])
            .where(
              (id) =>
                  (_result?.confidences[id] ?? 0) >=
                      _options.minDisplayConfidence ||
                  (_result?.fallingOrder.contains(id) ?? false),
            )
            .toList()
          ..sort(_compareTrackNames);
    final targetColors = _result?.targetColors ?? const <String, int>{};
    final progress = _progress;

    final children = <Widget>[
      if (!compact)
        const SectionHeader(title: 'Results', trailing: 'Scoring')
      else
        const Text(
          'Results',
          style: TextStyle(
            color: textPrimary,
            fontSize: 14,
            fontWeight: FontWeight.w800,
          ),
        ),
      if (compact) const SizedBox(height: 10),
      _ResultSummary(result: _result, compact: compact),
      SizedBox(height: compact ? 10 : 14),
      if (_state == _AppState.processing && progress != null)
        _ProcessingStatus(progress: progress),
      if (_state == _AppState.processing && progress != null)
        SizedBox(height: compact ? 10 : 14),
      _CleanTargetsPanel(
        targets: trackedTargets,
        confidences: _result?.confidences ?? const {},
        colors: targetColors,
      ),
      SizedBox(height: compact ? 10 : 14),

      if (_result != null && _result!.fallEvents.isNotEmpty) ...[
        const Text(
          'Fall positions',
          style: TextStyle(
            color: textSecondary,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.4,
          ),
        ),
        const SizedBox(height: 6),
        FloorMapPanel(
          fallEvents: _result!.fallEvents,
          targetColors: targetColors,
          originalWidth: _result!.originalWidth,
          originalHeight: _result!.originalHeight,
        ),
        SizedBox(height: compact ? 10 : 14),
      ],

      if (_options.debugMode &&
          (_liveCounts.isNotEmpty ||
              (_result?.countTimeline.isNotEmpty ?? false))) ...[
        const Text(
          'Visible objects over time',
          style: TextStyle(
            color: textSecondary,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.4,
          ),
        ),
        const SizedBox(height: 6),
        CountChart(
          timeline: _result?.countTimeline ?? _liveCounts,
          colors: () {
            final samples = _result?.countTimeline ?? _liveCounts;
            final keys = <String>{};
            for (final s in samples) {
              keys.addAll(s.counts.keys);
            }
            return {for (final k in keys) k: resolveColor(k, targetColors)};
          }(),
        ),
        SizedBox(height: compact ? 10 : 14),
      ],

      if (_options.debugMode &&
          (_liveLog.isNotEmpty || (_result?.eventLog.isNotEmpty ?? false))) ...[
        Row(
          children: [
            const Text(
              'Detection log',
              style: TextStyle(
                color: textSecondary,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.4,
              ),
            ),
            const Spacer(),
            Text(
              '${(_state == _AppState.processing ? _liveLog.length : _result?.eventLog.length ?? 0)} events',
              style: const TextStyle(color: textMuted, fontSize: 10),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ConstrainedBox(
          constraints: BoxConstraints(maxHeight: compact ? 140 : 180),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.18),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
            child: DetectionLog(
              events: _state == _AppState.processing
                  ? _liveLog
                  : (_result?.eventLog ?? const []),
              targetColors: targetColors,
            ),
          ),
        ),
        SizedBox(height: compact ? 10 : 14),
      ],

      Row(
        children: [
          Expanded(
            child: _SummaryTile(
              label: 'Motion Path',
              child: Text(
                _result == null
                    ? '—'
                    : (_result!.pathDrawn
                          ? '${_result!.carPathPoints} points'
                          : 'not detected'),
                style: const TextStyle(
                  color: textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: _SummaryTile(
              label: 'Processing',
              child: Text(
                _result?.processingTimeString ??
                    (_state == _AppState.processing ? 'running' : '—'),
                style: const TextStyle(
                  color: textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
        ],
      ),
    ];

    if (compact) {
      return ListView(
        padding: EdgeInsets.zero,
        children: [
          ...children,
          if (_state == _AppState.error && _errorMessage != null) ...[
            const SizedBox(height: 8),
            Text(
              _errorMessage!.split('\n').first,
              style: const TextStyle(color: error, fontSize: 11),
            ),
          ],
        ],
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ...children,
        const Spacer(),
        if (_state == _AppState.error && _errorMessage != null) ...[
          const SizedBox(height: 8),
          Text(
            _errorMessage!.split('\n').first,
            style: const TextStyle(color: error, fontSize: 11),
          ),
        ],
      ],
    );
  }
}

class _SelectedClipCard extends StatelessWidget {
  const _SelectedClipCard({
    required this.name,
    required this.state,
    required this.progress,
    required this.stage,
  });

  final String name;
  final _AppState state;
  final double progress;
  final PipelineStage? stage;

  @override
  Widget build(BuildContext context) {
    final status = switch (state) {
      _AppState.idle => '',
      _AppState.selected => 'ready',
      _AppState.processing => stage?.name ?? 'processing',
      _AppState.done => 'done',
      _AppState.error => 'error',
    };
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.20),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: accentPrimary.withValues(alpha: 0.13),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(
                  Icons.movie_filter_rounded,
                  color: accentPrimary,
                ),
              ),
              const SizedBox(width: 11),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      name,
                      style: const TextStyle(
                        color: textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 3),
                    Text(
                      status,
                      style: TextStyle(
                        color: state == _AppState.error ? error : textMuted,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (state == _AppState.processing) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(999),
              child: LinearProgressIndicator(
                minHeight: 6,
                value: progress.clamp(0.0, 1.0),
                backgroundColor: Colors.white.withValues(alpha: 0.08),
                valueColor: const AlwaysStoppedAnimation(accentPrimary),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CommandTile extends StatelessWidget {
  const _CommandTile({
    required this.label,
    required this.icon,
    required this.enabled,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final bool enabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: enabled ? onTap : null,
      borderRadius: BorderRadius.circular(18),
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 160),
        opacity: enabled ? 1 : 0.35,
        child: Container(
          height: 72,
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.045),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: accentSecondary, size: 22),
              const SizedBox(height: 7),
              Text(
                label,
                style: const TextStyle(
                  color: textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultBadge extends StatelessWidget {
  const _ResultBadge({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: accentSecondary, size: 16),
          const SizedBox(width: 7),
          Text(
            '$label ',
            style: const TextStyle(
              color: textMuted,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: textPrimary,
              fontSize: 12,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }
}

class _ProcessingStatus extends StatelessWidget {
  const _ProcessingStatus({required this.progress});

  final PipelineProgress progress;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.035),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(
                Icons.auto_awesome_motion_rounded,
                color: accentSecondary,
                size: 16,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  progress.message,
                  style: const TextStyle(color: textSecondary, fontSize: 11),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text(
                '${(progress.progress * 100).round()}%',
                style: const TextStyle(
                  color: textPrimary,
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              minHeight: 5,
              value: progress.progress.clamp(0.0, 1.0),
              backgroundColor: Colors.white.withValues(alpha: 0.06),
              valueColor: const AlwaysStoppedAnimation(accentPrimary),
            ),
          ),
        ],
      ),
    );
  }
}

class _CleanTargetsPanel extends StatelessWidget {
  const _CleanTargetsPanel({
    required this.targets,
    required this.confidences,
    required this.colors,
  });

  final List<String> targets;
  final Map<String, double> confidences;
  final Map<String, int> colors;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.035),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Text(
                'Targets',
                style: TextStyle(
                  color: textSecondary,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Spacer(),
              Text(
                '${targets.length}',
                style: const TextStyle(
                  color: textPrimary,
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (targets.isEmpty)
            const Text(
              'No confirmed targets yet',
              style: TextStyle(color: textMuted, fontSize: 11),
            )
          else
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final id in targets)
                  _TargetPill(
                    label: displayName(id),
                    confidence: confidences[id] ?? 0,
                    color: resolveColor(id, colors),
                  ),
              ],
            ),
        ],
      ),
    );
  }
}

class _TargetPill extends StatelessWidget {
  const _TargetPill({
    required this.label,
    required this.confidence,
    required this.color,
  });

  final String label;
  final double confidence;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(9),
        border: Border.all(color: color.withValues(alpha: 0.55)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: const TextStyle(
              color: textPrimary,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '${(confidence * 100).round()}%',
            style: const TextStyle(
              color: textSecondary,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _SmallActionButton extends StatelessWidget {
  const _SmallActionButton({
    required this.label,
    required this.icon,
    required this.enabled,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final bool enabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: AnimatedOpacity(
        duration: const Duration(milliseconds: 150),
        opacity: enabled ? 1 : 0.38,
        child: Container(
          height: 40,
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.045),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: textSecondary, size: 17),
              const SizedBox(width: 7),
              Text(
                label,
                style: const TextStyle(
                  color: textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResultSummary extends StatelessWidget {
  const _ResultSummary({required this.result, this.compact = false});

  final PipelineResult? result;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final r = result;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.035),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Row(
        children: [
          ScoreRing(
            score: r?.score ?? 0,
            total: r?.totalTargets ?? 0,
            size: compact ? 52 : 62,
          ),
          SizedBox(width: compact ? 10 : 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Final score',
                  style: TextStyle(
                    color: textMuted,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  r == null ? 'Run analysis' : '${r.score} / ${r.totalTargets}',
                  style: TextStyle(
                    color: textPrimary,
                    fontSize: compact ? 16 : 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  r == null
                      ? 'Order and path will appear here.'
                      : 'Order: ${r.fallingOrderString}',
                  style: const TextStyle(color: textSecondary, fontSize: 11),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 3),
                Text(
                  r == null
                      ? 'Path: —'
                      : 'Path: ${r.pathDrawn ? "drawn" : "not detected"}',
                  style: const TextStyle(color: textMuted, fontSize: 10),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _InputFileTile extends StatelessWidget {
  const _InputFileTile({
    required this.name,
    required this.state,
    required this.progress,
    required this.stage,
  });
  final String name;
  final _AppState state;
  final double progress;
  final PipelineStage? stage;

  @override
  Widget build(BuildContext context) {
    final pct = (progress * 100).clamp(0, 100).toStringAsFixed(0);
    final statusText = switch (state) {
      _AppState.idle => '',
      _AppState.selected => 'Ready',
      _AppState.processing => stage?.name ?? 'Processing',
      _AppState.done => 'Complete',
      _AppState.error => 'Error',
    };
    final statusColor = switch (state) {
      _AppState.done => success,
      _AppState.processing => accentPrimary,
      _AppState.error => error,
      _ => textMuted,
    };
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  gradient: gradientCool,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.movie_rounded,
                  color: Colors.white,
                  size: 18,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      name,
                      style: const TextStyle(
                        color: textPrimary,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    Text(
                      statusText.isEmpty ? '—' : statusText,
                      style: TextStyle(color: statusColor, fontSize: 10),
                    ),
                  ],
                ),
              ),
              if (state == _AppState.done)
                const Icon(
                  Icons.check_circle_rounded,
                  color: success,
                  size: 18,
                ),
            ],
          ),
          if (state == _AppState.processing) ...[
            const SizedBox(height: 6),
            Row(
              children: [
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      minHeight: 4,
                      value: progress,
                      backgroundColor: Colors.white.withValues(alpha: 0.06),
                      valueColor: const AlwaysStoppedAnimation(accentPrimary),
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  '$pct%',
                  style: const TextStyle(
                    color: accentPrimary,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _SummaryTile extends StatelessWidget {
  const _SummaryTile({required this.label, required this.child});
  final String label;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: const TextStyle(color: textMuted, fontSize: 10),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 6),
          child,
        ],
      ),
    );
  }
}

int _compareTrackNames(String a, String b) {
  if (a == 'rc_car') return 1;
  if (b == 'rc_car') return -1;
  final at = _targetNumber(a);
  final bt = _targetNumber(b);
  if (at != null && bt != null) return at.compareTo(bt);
  if (at != null) return -1;
  if (bt != null) return 1;
  final ap = a.split('#');
  final bp = b.split('#');
  final base = ap.first.compareTo(bp.first);
  if (base != 0) return base;
  final ai = ap.length > 1 ? int.tryParse(ap[1]) ?? 0 : 0;
  final bi = bp.length > 1 ? int.tryParse(bp[1]) ?? 0 : 0;
  return ai.compareTo(bi);
}

int? _targetNumber(String id) {
  if (!id.startsWith('target_')) return null;
  return int.tryParse(id.substring('target_'.length));
}
