# Hornet Detection AI

## Overview

This project focuses on building an AI model capable of detecting hornets using computer vision techniques. The core model is based on **YOLOv5** (You Only Look Once) architecture for object detection. The goal is to effectively identify and locate hornets in real-time video streams.

The initial experiment (Exp1) involves training the model using videos contained within the project. Specifically, the video data provided in the `Exp1` folder plays a crucial role in demonstrating the capabilities of the model.

## How It Works

1. **Dataset**: 
   - The training dataset comprises video footage in the `Exp1` folder, which contains labeled instances of hornets. The data is processed and fed into the YOLOv5 model for training.

2. **Model Architecture**:
   - The project employs YOLOv5, a state-of-the-art deep learning model for object detection tasks. YOLOv5 offers a balance between detection speed and accuracy, making it ideal for real-time detection of hornets.

3. **Training**:
   - Training is conducted using labeled frames extracted from the videos. The labeled dataset includes bounding boxes around hornets in various environmental conditions.

4. **Inference**:
   - Once the model is trained, it can be deployed to detect hornets in live video streams or other video inputs. The output includes bounding boxes around the detected hornets, along with confidence scores.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv5 weights:
   You can download the pre-trained YOLOv5 weights or train the model from scratch using the provided dataset in `Exp1`.

## Running the Detection

1. **Training**:
   To train the model on the provided video data, use the following command:
   ```bash
   python train.py --data data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --epochs 100
   ```

2. **Inference**:
   To run the hornet detection on new videos, run:
   ```bash
   python detect.py --source <video_path> --weights best.pt
   ```

## Results

The detection results for the initial experiment (Exp1) can be visualized in the provided video. In this experiment, the model was able to detect hornets with [X]% accuracy and Y% precision on the test dataset.

## Future Work

- Enhance the model's accuracy by expanding the dataset to include more diverse environments.
- Explore the integration of the AI into real-time video surveillance systems to monitor hornet activity in outdoor environments.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
