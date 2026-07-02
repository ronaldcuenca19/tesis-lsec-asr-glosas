import 'dart:convert';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:video_player/video_player.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Audio a Señas',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const AudioToSignPage(),
    );
  }
}

class AudioToSignPage extends StatefulWidget {
  const AudioToSignPage({super.key});

  @override
  State<AudioToSignPage> createState() => _AudioToSignPageState();
}

class _AudioToSignPageState extends State<AudioToSignPage> {
  // ============================
  // PON AQUI TU BACKEND
  // ============================
  // Copia aqui la URL ACTUAL que muestra cloudflared, sin espacios.
  // Los dominios trycloudflare.com suelen cambiar al reiniciar el tunel.
  static const String baseUrl =
      'https://badly-rob-teachers-salad.trycloudflare.com/';

  // Cambia este endpoint si tu backend usa otro nombre.
  static const String audioEndpoint = '/transcribe_audio_to_sign';

  String get _cleanBaseUrl => baseUrl.trim().replaceAll(RegExp(r'/+$'), '');

  Uri _buildBackendUri(String endpoint) {
    final cleanEndpoint = endpoint.trim();
    final normalizedEndpoint = cleanEndpoint.startsWith('/')
        ? cleanEndpoint
        : '/$cleanEndpoint';
    final value = '$_cleanBaseUrl$normalizedEndpoint';
    final uri = Uri.parse(value);

    if ((uri.scheme != 'http' && uri.scheme != 'https') || uri.host.isEmpty) {
      throw FormatException('URL del backend invalida: $value');
    }

    return uri;
  }

  String _resolveBackendUrl(String value) {
    final cleanValue = value.trim();

    if (cleanValue.isEmpty) {
      throw const FormatException(
        'El backend devolvio una URL de video vacia.',
      );
    }

    final possibleAbsoluteUri = Uri.tryParse(cleanValue);

    if (possibleAbsoluteUri != null &&
        (possibleAbsoluteUri.scheme == 'http' ||
            possibleAbsoluteUri.scheme == 'https') &&
        possibleAbsoluteUri.host.isNotEmpty) {
      return possibleAbsoluteUri.toString();
    }

    return Uri.parse('$_cleanBaseUrl/').resolve(cleanValue).toString();
  }

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _selectedAudioPlayer = AudioPlayer();

  VideoPlayerController? _resultVideoController;

  PlatformFile? _selectedAudioFile;
  Uint8List? _selectedAudioBytes;
  String? _selectedAudioPath;
  String? _selectedAudioName;

  bool _isRecording = false;
  bool _isLoading = false;
  bool _isResultVideoFinished = false;

  String? _errorMessage;

  List<dynamic> _plan = [];
  List<dynamic> _missing = [];

  String? _videoUrl;
  String? _recognizedText;
  String? _glosses;

  @override
  void dispose() {
    _selectedAudioPlayer.dispose();
    _recorder.dispose();

    _resultVideoController?.removeListener(_onResultVideoTick);
    _resultVideoController?.dispose();

    super.dispose();
  }

  Future<void> _clearResult() async {
    _resultVideoController?.removeListener(_onResultVideoTick);
    await _resultVideoController?.dispose();

    _resultVideoController = null;

    _plan = [];
    _missing = [];
    _videoUrl = null;
    _recognizedText = null;
    _glosses = null;
    _errorMessage = null;
    _isResultVideoFinished = false;
  }

  void _onResultVideoTick() {
    final controller = _resultVideoController;

    if (controller == null || !controller.value.isInitialized) return;

    final duration = controller.value.duration;
    final position = controller.value.position;

    if (duration == Duration.zero) return;

    final isFinished =
        position >= duration - const Duration(milliseconds: 200) &&
        !controller.value.isPlaying;

    if (isFinished != _isResultVideoFinished && mounted) {
      setState(() {
        _isResultVideoFinished = isFinished;
      });
    }
  }

  Future<void> _prepareAudioPreview() async {
    try {
      await _selectedAudioPlayer.stop();

      final path = _selectedAudioPath;

      if (path == null || path.isEmpty) {
        return;
      }

      if (kIsWeb) {
        await _selectedAudioPlayer.setUrl(path);
      } else {
        await _selectedAudioPlayer.setFilePath(path);
      }

      await _selectedAudioPlayer.seek(Duration.zero);
    } catch (e) {
      setState(() {
        _errorMessage = 'No se pudo preparar la vista previa del audio: $e';
      });
    }
  }

  Future<void> _pickAudio() async {
    try {
      await _selectedAudioPlayer.stop();
      await _clearResult();

      final result = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'],
        allowMultiple: false,
        withData: kIsWeb,
      );

      if (result == null || result.files.isEmpty) return;

      final file = result.files.single;

      _selectedAudioFile = file;
      _selectedAudioBytes = file.bytes;
      _selectedAudioName = file.name;
      _selectedAudioPath = file.path;

      if (!mounted) return;
      setState(() {});

      await _prepareAudioPreview();

      if (!mounted) return;
      setState(() {});
    } catch (e) {
      setState(() {
        _errorMessage = 'No se pudo cargar el audio: $e';
      });
    }
  }

  Future<void> _startRecording() async {
    try {
      await _selectedAudioPlayer.stop();
      await _clearResult();

      final hasPermission = await _recorder.hasPermission();

      if (!hasPermission) {
        setState(() {
          _errorMessage = 'No se concedió permiso para usar el micrófono.';
        });
        return;
      }

      final timestamp = DateTime.now().millisecondsSinceEpoch;

      String path;

      if (kIsWeb) {
        path = 'grabacion_$timestamp.wav';
      } else {
        final dir = await getTemporaryDirectory();
        path = '${dir.path}/grabacion_$timestamp.wav';
      }

      const config = RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: 16000,
        numChannels: 1,
        echoCancel: true,
        noiseSuppress: true,
        autoGain: true,
      );

      await _recorder.start(config, path: path);

      setState(() {
        _isRecording = true;
        _selectedAudioFile = null;
        _selectedAudioBytes = null;
        _selectedAudioPath = null;
        _selectedAudioName = null;
        _errorMessage = null;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'No se pudo iniciar la grabación: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      final path = await _recorder.stop();

      if (path == null || path.isEmpty) {
        throw Exception('No se obtuvo la ruta del audio grabado.');
      }

      final generatedName = path.split('/').last.isEmpty
          ? 'grabacion_audio.wav'
          : path.split('/').last;

      setState(() {
        _isRecording = false;
        _selectedAudioFile = null;
        _selectedAudioBytes = null;
        _selectedAudioPath = path;
        _selectedAudioName = generatedName;
        _errorMessage = null;
      });

      await _prepareAudioPreview();

      if (!mounted) return;
      setState(() {});
    } catch (e) {
      setState(() {
        _isRecording = false;
        _errorMessage = 'No se pudo detener la grabación: $e';
      });
    }
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _sendAudio() async {
    if (_selectedAudioBytes == null &&
        (_selectedAudioPath == null || _selectedAudioPath!.isEmpty)) {
      setState(() {
        _errorMessage = 'Primero selecciona o graba un audio.';
      });
      return;
    }

    if (_isRecording) {
      setState(() {
        _errorMessage = 'Detén la grabación antes de enviar el audio.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _plan = [];
      _missing = [];
      _videoUrl = null;
      _recognizedText = null;
      _glosses = null;
      _isResultVideoFinished = false;
    });

    try {
      final uri = _buildBackendUri(audioEndpoint);
      final request = http.MultipartRequest('POST', uri);

      final filename = _selectedAudioName ?? 'audio.wav';

      if (_selectedAudioBytes != null) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'audio',
            _selectedAudioBytes!,
            filename: filename,
          ),
        );
      } else {
        if (kIsWeb) {
          throw Exception(
            'En Flutter Web no se pudo obtener bytes del audio. '
            'Selecciona un archivo con soporte de bytes o prueba en Android.',
          );
        }

        request.files.add(
          await http.MultipartFile.fromPath(
            'audio',
            _selectedAudioPath!,
            filename: filename,
          ),
        );
      }

      final streamedResponse = await request.send();
      final responseBody = await streamedResponse.stream.bytesToString();

      if (streamedResponse.statusCode != 200) {
        String message = 'Error ${streamedResponse.statusCode}';

        try {
          final decoded = jsonDecode(responseBody);
          if (decoded is Map && decoded['detail'] != null) {
            message = decoded['detail'].toString();
          }
        } catch (_) {}

        throw Exception(message);
      }

      final data = jsonDecode(responseBody) as Map<String, dynamic>;

      final rawVideoUrl = (data['video_url'] ?? '').toString().trim();

      if (rawVideoUrl.isEmpty) {
        throw Exception('El backend no devolvió video_url.');
      }

      final fullVideoUrl = _resolveBackendUrl(rawVideoUrl);

      _plan = (data['plan'] as List?) ?? [];
      _missing = (data['missing'] as List?) ?? [];

      _recognizedText =
          (data['text'] ?? data['transcription'] ?? data['transcript'])
              ?.toString();

      _glosses = (data['glosses'] ?? data['glosas'] ?? data['gloss_text'])
          ?.toString();

      _videoUrl = fullVideoUrl;

      _resultVideoController?.removeListener(_onResultVideoTick);
      await _resultVideoController?.dispose();

      _resultVideoController = VideoPlayerController.networkUrl(
        Uri.parse(fullVideoUrl),
      );

      await _resultVideoController!.initialize();

      // IMPORTANTE:
      // El video resultado ya NO se reproduce en bucle.
      await _resultVideoController!.setLooping(false);

      _isResultVideoFinished = false;

      _resultVideoController!.addListener(_onResultVideoTick);

      if (!mounted) return;
      setState(() {});
    } catch (e) {
      setState(() {
        _errorMessage = 'Error al procesar el audio: $e';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildAudioPreview() {
    final hasAudio =
        _selectedAudioBytes != null ||
        (_selectedAudioPath != null && _selectedAudioPath!.isNotEmpty);

    if (!hasAudio) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Audio seleccionado / grabado',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text('Nombre: ${_selectedAudioName ?? 'audio.wav'}'),
            if (_selectedAudioPath != null) ...[
              const SizedBox(height: 4),
              Text(_selectedAudioPath!, style: const TextStyle(fontSize: 12)),
            ],
            const SizedBox(height: 12),
            StreamBuilder<PlayerState>(
              stream: _selectedAudioPlayer.playerStateStream,
              builder: (context, snapshot) {
                final playerState = snapshot.data;
                final isPlaying = playerState?.playing ?? false;

                return Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton.filled(
                      onPressed: () async {
                        try {
                          if (isPlaying) {
                            await _selectedAudioPlayer.pause();
                          } else {
                            await _selectedAudioPlayer.play();
                          }
                        } catch (e) {
                          setState(() {
                            _errorMessage =
                                'No se pudo reproducir el audio: $e';
                          });
                        }
                      },
                      icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                      tooltip: isPlaying ? 'Pausar audio' : 'Reproducir audio',
                    ),
                    const SizedBox(width: 12),
                    IconButton.outlined(
                      onPressed: () async {
                        await _selectedAudioPlayer.seek(Duration.zero);
                        await _selectedAudioPlayer.pause();
                      },
                      icon: const Icon(Icons.stop),
                      tooltip: 'Detener audio',
                    ),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _openResultVideoFullscreen(
    VideoPlayerController controller,
  ) async {
    if (!controller.value.isInitialized || !mounted) return;

    await Navigator.of(context).push<void>(
      MaterialPageRoute<void>(
        fullscreenDialog: true,
        builder: (_) => FullscreenSignVideoPage(controller: controller),
      ),
    );

    // Actualiza los controles del reproductor pequeño al regresar.
    if (mounted) {
      setState(() {});
    }
  }

  Widget _buildVideoPlayer(VideoPlayerController controller) {
    if (!controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Column(
      children: [
        Semantics(
          button: true,
          label: 'Abrir el video de señas en pantalla completa horizontal',
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: () => _openResultVideoFullscreen(controller),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: AspectRatio(
                aspectRatio: controller.value.aspectRatio == 0
                    ? 16 / 9
                    : controller.value.aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    ColoredBox(
                      color: Colors.black,
                      child: VideoPlayer(controller),
                    ),
                    Positioned(
                      top: 8,
                      right: 8,
                      child: IgnorePointer(
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: const BoxDecoration(
                            color: Colors.black54,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.fullscreen,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      left: 8,
                      bottom: 8,
                      child: IgnorePointer(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text(
                            'Toca el video para ampliar',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),

        // Botones normales: continuar y pausar.
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            IconButton.filled(
              onPressed: () async {
                if (_isResultVideoFinished) {
                  await controller.seekTo(Duration.zero);
                  setState(() {
                    _isResultVideoFinished = false;
                  });
                }

                await controller.play();
                setState(() {});
              },
              icon: const Icon(Icons.play_arrow),
              tooltip: 'Continuar',
            ),
            const SizedBox(width: 12),
            IconButton.outlined(
              onPressed: controller.value.isPlaying
                  ? () async {
                      await controller.pause();
                      setState(() {});
                    }
                  : null,
              icon: const Icon(Icons.pause),
              tooltip: 'Pausar',
            ),
          ],
        ),

        // Este botón solo aparece cuando el video ya terminó.
        if (_isResultVideoFinished) ...[
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: () async {
              await controller.seekTo(Duration.zero);

              setState(() {
                _isResultVideoFinished = false;
              });

              await controller.play();
              setState(() {});
            },
            icon: const Icon(Icons.replay),
            label: const Text('Reproducir de nuevo'),
          ),
        ],
      ],
    );
  }

  Widget _buildTextResults() {
    if ((_recognizedText == null || _recognizedText!.isEmpty) &&
        (_glosses == null || _glosses!.isEmpty)) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_recognizedText != null && _recognizedText!.isNotEmpty) ...[
          const Text(
            'Texto reconocido',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          SelectableText(_recognizedText!),
          const SizedBox(height: 16),
        ],
      ],
    );
  }

  Widget _buildMissing() {
    if (_missing.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Missing',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        ..._missing.map((item) {
          if (item is! Map) {
            return Card(child: ListTile(title: Text(item.toString())));
          }

          final map = Map<String, dynamic>.from(item);

          return Card(
            color: Colors.orange.shade50,
            child: ListTile(
              title: Text('${map['source_gloss'] ?? ''}'),
              subtitle: Text(
                'unit: ${map['unit'] ?? ''}\nmode: ${map['mode'] ?? ''}',
              ),
              isThreeLine: true,
            ),
          );
        }),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final resultController = _resultVideoController;

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          'Traductor de lenguaje de señas',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        backgroundColor: Colors.indigo,
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Transcripción de la voz a lenguaje de señas'),
            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isLoading || _isRecording ? null : _pickAudio,
                    icon: const Icon(Icons.audio_file),
                    label: const Text('Cargar audio'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isLoading ? null : _toggleRecording,
                    icon: Icon(_isRecording ? Icons.stop_circle : Icons.mic),
                    label: Text(_isRecording ? 'Detener' : 'Grabar'),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _isLoading || _isRecording ? null : _sendAudio,
                icon: const Icon(Icons.check),
                label: const Text('Procesar audio'),
              ),
            ),

            const SizedBox(height: 20),

            if (_isRecording) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.indigo.shade50,
                  border: Border.all(color: Colors.indigo.shade200),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.mic, color: Colors.indigo),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Grabando audio... presiona “Detener” para finalizar.',
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
            ],

            _buildAudioPreview(),

            const SizedBox(height: 20),

            if (_isLoading) ...[
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 12),
              const Center(child: Text('Procesando audio...')),
              const SizedBox(height: 24),
            ],

            if (_errorMessage != null) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  border: Border.all(color: Colors.red.shade200),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.red),
                ),
              ),
              const SizedBox(height: 24),
            ],

            _buildTextResults(),

            if (_videoUrl != null && resultController != null) ...[
              const Text(
                'Video resultado',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              _buildVideoPlayer(resultController),
              const SizedBox(height: 24),
            ],

            const SizedBox(height: 24),
            _buildMissing(),
          ],
        ),
      ),
    );
  }
}

class FullscreenSignVideoPage extends StatefulWidget {
  const FullscreenSignVideoPage({super.key, required this.controller});

  final VideoPlayerController controller;

  @override
  State<FullscreenSignVideoPage> createState() =>
      _FullscreenSignVideoPageState();
}

class _FullscreenSignVideoPageState extends State<FullscreenSignVideoPage> {
  bool _isVideoFinished = false;
  bool _isRestarting = false;
  bool _systemUiRestored = false;

  bool get _canControlOrientation {
    if (kIsWeb) return false;

    return defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onVideoChanged);
    _initializeFullscreenVideo();
  }

  bool _isNearVideoEnd(VideoPlayerValue value) {
    if (!value.isInitialized || value.duration == Duration.zero) {
      return false;
    }

    return value.position >= value.duration - const Duration(milliseconds: 250);
  }

  bool _hasVideoFinished(VideoPlayerValue value) {
    return _isNearVideoEnd(value) && !value.isPlaying;
  }

  void _onVideoChanged() {
    // Durante el reinicio, el controlador puede informar durante unos
    // milisegundos la posición anterior (el final del video). Ignoramos ese
    // estado transitorio para que el botón no reaparezca y bloquee la vista.
    if (_isRestarting) return;

    final finished = _hasVideoFinished(widget.controller.value);

    if (finished != _isVideoFinished && mounted) {
      setState(() {
        _isVideoFinished = finished;
      });
    }
  }

  Future<void> _initializeFullscreenVideo() async {
    await _enterFullscreenMode();

    if (!mounted) return;

    await _playAutomatically();
  }

  Future<void> _playAutomatically() async {
    final controller = widget.controller;

    if (!controller.value.isInitialized) return;

    // Si se abre la pantalla completa cuando el video ya estaba al final,
    // se aplica el mismo reinicio seguro utilizado por el botón de repetición.
    if (_isNearVideoEnd(controller.value)) {
      await _restartVideoFromBeginning();
      return;
    }

    if (mounted) {
      setState(() {
        _isVideoFinished = false;
      });
    }

    await controller.play();
  }

  Future<void> _waitUntilPositionIsAtStart(
    VideoPlayerController controller,
  ) async {
    // En algunos dispositivos Android, seekTo() termina antes de que el
    // reproductor nativo publique la nueva posición. Esperamos brevemente.
    for (var attempt = 0; attempt < 12; attempt++) {
      if (controller.value.position <= const Duration(milliseconds: 250)) {
        return;
      }

      await Future<void>.delayed(const Duration(milliseconds: 40));
    }
  }

  Future<bool> _waitUntilPlaybackStarts(
    VideoPlayerController controller,
  ) async {
    final initialPosition = controller.value.position;

    for (var attempt = 0; attempt < 20; attempt++) {
      await Future<void>.delayed(const Duration(milliseconds: 50));

      final value = controller.value;

      if (value.hasError) return false;

      final positionAdvanced =
          value.position > initialPosition + const Duration(milliseconds: 20);

      if (value.isPlaying || positionAdvanced) {
        return true;
      }
    }

    return false;
  }

  Future<void> _restartVideoFromBeginning() async {
    if (_isRestarting) return;

    final controller = widget.controller;

    if (!controller.value.isInitialized) return;

    _isRestarting = true;

    // El botón desaparece antes de iniciar la operación para mantener limpia
    // la pantalla durante la repetición.
    if (mounted) {
      setState(() {
        _isVideoFinished = false;
      });
    }

    var playbackStarted = false;

    try {
      // Pausar antes de seekTo evita que Android conserve el estado EOS
      // (end of stream) y se quede mostrando congelado el último fotograma.
      await controller.pause();
      await controller.seekTo(Duration.zero);
      await _waitUntilPositionIsAtStart(controller);

      // Dejamos que el reproductor nativo procese el seek antes de play().
      await Future<void>.delayed(const Duration(milliseconds: 80));
      await controller.play();
      playbackStarted = await _waitUntilPlaybackStarts(controller);

      // Reintento controlado para ciertos dispositivos donde el primer play()
      // después de finalizar el archivo no abandona correctamente el estado EOS.
      if (!playbackStarted) {
        await controller.pause();
        await Future<void>.delayed(const Duration(milliseconds: 80));
        await controller.seekTo(Duration.zero);
        await _waitUntilPositionIsAtStart(controller);
        await Future<void>.delayed(const Duration(milliseconds: 120));
        await controller.play();
        playbackStarted = await _waitUntilPlaybackStarts(controller);
      }
    } catch (_) {
      playbackStarted = false;
    } finally {
      _isRestarting = false;

      if (mounted) {
        setState(() {
          // Si ambos intentos fallan, el botón vuelve a aparecer para que el
          // usuario pueda intentarlo otra vez en lugar de dejar la vista bloqueada.
          _isVideoFinished = !playbackStarted;
        });
      }
    }
  }

  Future<void> _replayVideo() async {
    await _restartVideoFromBeginning();
  }

  Future<void> _enterFullscreenMode() async {
    if (!_canControlOrientation) return;

    try {
      await SystemChrome.setPreferredOrientations(const [
        DeviceOrientation.landscapeLeft,
        DeviceOrientation.landscapeRight,
      ]);

      if (_systemUiRestored) return;

      await SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    } catch (_) {
      // La reproducción continúa aunque el sistema no permita bloquear
      // la orientación o esconder completamente sus barras.
    }
  }

  Future<void> _restorePortraitMode() async {
    if (_systemUiRestored || !_canControlOrientation) return;
    _systemUiRestored = true;

    try {
      await SystemChrome.setPreferredOrientations(const [
        DeviceOrientation.portraitUp,
      ]);
      await SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    } catch (_) {
      // No impide el cierre si el sistema rechaza el cambio de orientación.
    }
  }

  Future<void> _closeFullscreen() async {
    await _restorePortraitMode();

    if (mounted) {
      Navigator.of(context).pop();
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onVideoChanged);
    _restorePortraitMode();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final value = controller.value;

    if (!value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final aspectRatio = value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio;
    final screenPadding = MediaQuery.of(context).padding;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        alignment: Alignment.center,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: aspectRatio,
              child: VideoPlayer(controller),
            ),
          ),

          // Se conserva únicamente un botón discreto para salir.
          Positioned(
            top: screenPadding.top + 8,
            left: screenPadding.left + 8,
            child: IconButton(
              onPressed: _closeFullscreen,
              style: IconButton.styleFrom(
                backgroundColor: Colors.black45,
                foregroundColor: Colors.white,
              ),
              icon: const Icon(Icons.close),
              tooltip: 'Salir de pantalla completa',
            ),
          ),

          // Este control aparece únicamente cuando el video termina.
          if (_isVideoFinished && !_isRestarting)
            Positioned.fill(
              child: ColoredBox(
                color: Colors.black38,
                child: Center(
                  child: FilledButton.icon(
                    onPressed: _replayVideo,
                    icon: const Icon(Icons.replay),
                    label: const Text('Reproducir de nuevo'),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
