# Spatio-Temporal Intelligence for Real-Time Vehicle Detection

## Description

This project implements a computer vision system to detect, track, and predict the movement of vehicles in video streams. It leverages a hybrid approach combining spatial object detection with temporal sequence modeling to provide a comprehensive analysis of vehicle motion. The system processes video input, identifies various vehicle types, assigns consistent tracking IDs, and uses the extracted trajectories to forecast future positions. This provides valuable spatio-temporal data applicable to intelligent traffic monitoring, flow analysis, and serves as foundational work for autonomous driving systems.

## Technologies Used

* **Programming Language:** Python
* **Environment:** Google Colab (leveraging GPU acceleration)
* **Core Libraries:**
    * `ultralytics`: For the YOLOv8 object detection and tracking model.
    * `TensorFlow / Keras`: For building and training the LSTM temporal model.
    * `OpenCV`: For video input/output and frame processing.
    * `NumPy`: For numerical operations.
    * `Matplotlib`: For plotting results (like the LSTM loss graph and prediction visualization).

## How to Run

1.  **Environment:** This project is designed to be run in a Google Colab notebook.
2.  **Dependencies:** Install the required library by running the following command in a Colab cell:
    ```bash
    !pip install ultralytics
    ```
3.  **Input Video:** Upload your desired input video file (e.g., in `.mp4` format) to your Colab session.
4.  **Update Filenames:** In the code cells that process the video (both for YOLO tracking and trajectory extraction), locate the `video_filename` variable and update its value to match the exact name of the video file you uploaded.
5.  **Execute Cells:** Run the Colab notebook cells sequentially.

## Modules & Results

The system consists of two main modules:

1.  **Spatial Module (YOLOv8 + BoT-SORT):**
    * Uses the pre-trained `yolov8n.pt` model with the BoT-SORT tracker (`model.track()`).
    * Detects and tracks multiple vehicle classes (cars, trucks, buses, motorbikes) in real-time.
    * Configured with a confidence threshold of 0.7 for improved accuracy.
    * **Result:** Generates an output video with stable bounding boxes and tracking IDs overlaid on the detected vehicles.

2.  **Temporal Module (LSTM):**
    * Extracts vehicle trajectories (sequences of bounding box coordinates) from the spatial module's output.
    * Uses a sliding window approach (sequence length: 10) to prepare data for training.
    * Trains an LSTM network (50 units) to predict the next bounding box based on the previous sequence.
    * **Result:** Successfully predicts future vehicle positions, as validated by low training/validation MSE loss and visual comparison of predicted vs. ground truth boxes.
