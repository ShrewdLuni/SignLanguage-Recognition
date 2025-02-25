# American Sign Language Recognizer

![Demo GIF](\assets\IMG_3358.gif)


## Overview

ASL Recognizer is a real-time hand gesture recognition system that utilizes computer vision and machine learning to recognize American Sign Language (ASL) Alphabet. The project leverages OpenCV, MediaPipe, and a trained model to identify hand gestures captured via a webcam.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe's hand tracking model.
- **Machine Learning Model**: Trained using a Random Forest Classifier.
- **Gesture Recognition**: Detects ASL letters from hand movements.
- **Live Prediction Display**: Shows real-time predictions and accumulates recognized letters into a word.

## Installation

### Prerequisites

Ensure you have Python installed (Python 3.x recommended). Install required dependencies using:

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

## Usage

### Run ASL Recognition  
Start real-time gesture recognition using the pretrained model:  

```bash
python app.py
```
The program will use the webcam to detect hand gestures and predict ASL letters, displaying them on the screen.

### Train Custom Module (Optional)
To train a model with your own signs, run:
```bash
python getModel.py
```

## Project Structure

```bash
SignLanguage/                # Root directory of the project
│── assets/                  # Directory for storing project assets
│   ├── data.pickle          # Preprocessed dataset used for training or inference
│   ├── model.p              # Pretrained machine learning model file
│── app.py                   # Main application script, runs the base model by default
│── getModel.py              # Script for training a custom model using user-specific signs
```

## License

This project is open-source and available under the MIT License.

