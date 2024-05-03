import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:permission_handler/permission_handler.dart';
import 'package:cat_dog_detector/result.dart';
import 'dart:math';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late File _image;
  dynamic _probability = 0;
  String? _result;
  List<String>? _labels;
  late tfl.Interpreter _interpreter;
  final picker = ImagePicker();
  late List<CameraDescription> cameras;
  late CameraController cameraController;
  bool _isRecording = false;
  String _sentence = '';
  int _countdown = 3;
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      loadLabels().then((loadedLabels) {
        setState(() {
          _labels = loadedLabels;
        });
      });
    });
    _checkCameraPermission();
  }

  @override
  void dispose() {
    _interpreter.close();
    cameraController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromARGB(255, 255, 187, 0),
      body: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: <Widget>[
            const SizedBox(height: 80),
            const Text(
              'Sign Language Detector App',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                height: 1.4,
                fontFamily: 'SofiaSans',
                fontSize: 30,
              ),
            ),
            const SizedBox(height: 50),
            Center(
              child: SizedBox(
                width: 350,
                child: Column(
                  children: <Widget>[
                    Image.asset('assets/cat_dog_icon.png'),
                    const SizedBox(height: 50),
                  ],
                ),
              ),
            ),
            SizedBox(
              width: MediaQuery.of(context).size.width,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: <Widget>[
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        pickImageFromCamera();
                      },
                      child: Container(
                        alignment: Alignment.center,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 18,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black38,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text(
                          'Capture a Photo',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'SofiaSans',
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 20),
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        pickImageFromGallery();
                      },
                      child: Container(
                        alignment: Alignment.center,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 18,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black38,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text(
                          'Select a photo',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'SofiaSans',
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 20),
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        // Navigate to the Live translation screen
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => MyHomePage()),
                        );
                      },
                      child: Container(
                        alignment: Alignment.center,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 18,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black38,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: const Text(
                          'Live Translation',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'SofiaSans',
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> loadModel() async {
    try {
      _interpreter =
          await tfl.Interpreter.fromAsset('assets/model_unquant.tflite');
    } catch (e) {
      debugPrint('Error loading model: $e');
    }
  }

  Future<void> pickImageFromCamera() async {
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      _setImage(File(pickedFile.path));
    }
  }

  Future<void> pickImageFromGallery() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      _setImage(File(pickedFile.path));
    }
  }

  void _setImage(File image) {
    setState(() {
      _image = image;
    });
    runInference();
  }

  Future<Uint8List> preprocessImage(File imageFile) async {
    // Decode the image to an Image object
    img.Image? originalImage =
        img.decodeImage(await imageFile.readAsBytes());

    // Resize the image to the correct size
    img.Image resizedImage =
        img.copyResize(originalImage!, width: 224, height: 224);

    // Convert to a byte buffer in the format suitable for TensorFlow Lite (RGB)
    // The model expects a 4D tensor [1, 224, 224, 3]
    // Flatten the resized image to match this shape
    Uint8List bytes = resizedImage.getBytes();
    return bytes;
  }

  Future<void> runInference() async {
    if (_labels == null) {
      return;
    }

    try {
      Uint8List inputBytes = await preprocessImage(_image);
      var input = inputBytes.buffer.asUint8List().reshape([1, 224, 224, 3]);
      var outputBuffer = List<double>.filled(1 * 16, 0).reshape([1, 16]);

      _interpreter.run(input, outputBuffer);

      // Assuming output is now List<List<int>> after inference
      List<double> output = outputBuffer[0];

      // Print raw output for debugging
      debugPrint('Raw output: $output');

      // Calculate probability
      double maxScore = output.reduce(max);
      _probability = (maxScore / 255.0); // Convert to percentage

      // Get the classification result
      int highestProbIndex = output.indexOf(maxScore);
      String classificationResult = _labels![highestProbIndex];

      setState(() {
        _result = classificationResult;
        // _probability is updated with the calculated probability
      });

      navigateToResult();
    } catch (e) {
      debugPrint('Error during inference: $e');
    }
  }

  Future<List<String>> loadLabels() async {
    final labelsData =
        await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    return labelsData.split('\n');
  }

  String classifyImage(List<int> output) {
    int highestProbIndex = output.indexOf(output.reduce(max));
    return _labels![highestProbIndex];
  }

  void navigateToResult() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(
          image: _image,
          result: _result!,
          probability: _probability,
        ),
      ),
    );
  }

  Future<void> _checkCameraPermission() async {
    PermissionStatus status = await Permission.camera.status;
    if (status.isGranted) {
      _initializeCamera();
    } else {
      await Permission.camera.request();
      if (await Permission.camera.isGranted) {
        _initializeCamera();
      } else {
        // Handle permission denied
        print('Camera permission denied');
      }
    }
  }

  Future<void> _initializeCamera() async {
    cameras = await availableCameras();
    cameraController = CameraController(
      cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );
    await cameraController.initialize().then((value) {
      if (!mounted) {
        return;
      }
      setState(() {});
    }).catchError((e) {
      print(e);
    });
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late File _image;
  dynamic _probability = 0;
  String? _result;
  List<String>? _labels;
  late tfl.Interpreter _interpreter;
  final picker = ImagePicker();
  late List<CameraDescription> cameras;
  late CameraController cameraController;
  bool _isRecording = false;
  String _sentence = '';
  int _countdown = 3;
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      loadLabels().then((loadedLabels) {
        setState(() {
          _labels = loadedLabels;
        });
      });
    });
    _checkCameraPermission();
  }
  Future<List<String>> loadLabels() async {
    final labelsData =
        await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    return labelsData.split('\n');
  }

  @override
  void dispose() {
    _interpreter.close();
    cameraController.dispose();
    super.dispose();
  }

   Future<void> loadModel() async {
    try {
      _interpreter =
          await tfl.Interpreter.fromAsset('assets/model_unquant.tflite');
    } catch (e) {
      debugPrint('Error loading model: $e');
    }
  }
  Future<Uint8List> preprocessImage(File imageFile) async {
    // Decode the image to an Image object
    img.Image? originalImage =
        img.decodeImage(await imageFile.readAsBytes());

    // Resize the image to the correct size
    img.Image resizedImage =
        img.copyResize(originalImage!, width: 224, height: 224);

    // Convert to a byte buffer in the format suitable for TensorFlow Lite (RGB)
    // The model expects a 4D tensor [1, 224, 224, 3]
    // Flatten the resized image to match this shape
    Uint8List bytes = resizedImage.getBytes();
    return bytes;
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Sign Language to Sentence'),
      ),
      body: Stack(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: cameraController.value.isInitialized
                    ? Container(
                        color: Colors.white,
                        child: AspectRatio(
                          aspectRatio: cameraController.value.aspectRatio,
                          child: CameraPreview(cameraController),
                        ),
                      )
                    : Center(child: CircularProgressIndicator()),
              ),
              SizedBox(height: 20), // Add space between text and buttons
              _buildTextInput(), // Add the disabled textbox
              SizedBox(height: 20), // Add space between textbox and buttons
              _buildControlButtons(),
            ],
          ),
          if (_isRecording && _countdown > 0)
            Center(
              child: Text(
                _countdown.toString(),
                style: TextStyle(
                  fontSize: 48,
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildTextInput() {
  return Padding(
    padding: const EdgeInsets.symmetric(horizontal: 20.0),
    child: TextField(
      enabled: false,
      decoration: InputDecoration(
        labelText: 'Translation Shows Here',
        border: OutlineInputBorder(),
        labelStyle: TextStyle(
          color: Colors.black, // Change label text color
        ),
      ),
      style: TextStyle(
        color: Colors.black, // Change text color
      ),
      controller: TextEditingController(text: _sentence), // Set text to _sentence
    ),
  );
}

  Widget _buildControlButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        ElevatedButton(
          onPressed: _isRecording ? null : _startRecording,
          child: Text('Start'),
        ),
        SizedBox(width: 20),
        ElevatedButton(
          onPressed: _isRecording ? _stopRecording : null,
          child: Text('Stop'),
        ),
      ],
    );
  }

 void _startRecording() {
  setState(() {
    _isRecording = true;
    _sentence = '';
    _countdown = 3; // Reset countdown to 3 when recording starts
  });
  _timer = Timer.periodic(Duration(seconds: 1), (timer) {
    setState(() {
      if (_countdown > 0) {
        _countdown--;
      } else {
        _countdown = 3; // Reset countdown to 3 when it reaches 0
        _takeAndProcessPicture(); // Take and process picture when countdown reaches 0
      }
    });
  });
  cameraController.startImageStream((CameraImage cameraImage) {});
}

  void _stopRecording() {
  cameraController.stopImageStream();
  _timer.cancel(); // Cancel the countdown timer when recording stops
  setState(() {
    _isRecording = false;
  });
  }
  
  Future<void> _takeAndProcessPicture() async {
  try {
    XFile imageFile = await cameraController.takePicture();
    if (imageFile != null) {
      File image = File(imageFile.path);
      await _processImage(image);
    }
  } catch (e) {
    print('Error taking picture: $e');
  }
}
Future<void> _processImage(File image) async {
  if (_interpreter == null || _labels == null) {
    return;
  }

  try {
    Uint8List inputBytes = await preprocessImage(image);
    var input = inputBytes.buffer.asUint8List().reshape([1, 224, 224, 3]);
    var outputBuffer = List<double>.filled(1 * 16, 0).reshape([1, 16]);

    _interpreter.run(input, outputBuffer);

    // Assuming output is now List<List<int>> after inference
    List<double> output = outputBuffer[0];

    // Print raw output for debugging
    debugPrint('Raw output: $output');

    // Calculate probability
    double maxScore = output.reduce(max);
    double probability = maxScore / 255.0; // Convert to percentage

    // Get the classification result
    int highestProbIndex = output.indexOf(maxScore);
    String classificationResult = _labels![highestProbIndex];

    setState(() {
      _sentence += classificationResult + ' '; // Append result to the sentence
    });
  } catch (e) {
    debugPrint('Error during inference: $e');
  }
}

  Future<void> _checkCameraPermission() async {
    PermissionStatus status = await Permission.camera.status;
    if (status.isGranted) {
      _initializeCamera();
    } else {
      await Permission.camera.request();
      if (await Permission.camera.isGranted) {
        _initializeCamera();
      } else {
        // Handle permission denied
        print('Camera permission denied');
      }
    }
  }

  Future<void> _initializeCamera() async {
    cameras = await availableCameras();
    cameraController = CameraController(
      cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );
    await cameraController.initialize().then((value) {
      if (!mounted) {
        return;
      }
      setState(() {});
    }).catchError((e) {
      print(e);
    });
  }
}
