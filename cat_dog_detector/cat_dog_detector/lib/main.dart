import 'package:cat_dog_detector/home.dart';
import 'package:flutter/material.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cat and Dog Classifer',
      home: HomePage(),
      debugShowCheckedModeBanner: false,
      theme:  ThemeData( primaryColor: const Color.fromARGB(211, 198, 184, 184), // Your primary color
  hintColor: Colors.black38, // Your additional accent color
      ),
    );
  }
}
