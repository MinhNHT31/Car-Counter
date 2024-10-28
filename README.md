# 🚗 Car Counting Project

## Overview

This project implements a car counting system using YOLOv10 for vehicle detection and ByteTrack for object tracking. The system is designed to monitor and count the number of vehicles passing through a designated area in real-time, making it valuable for traffic management and urban planning. 🌆

## Features

- **Real-Time Vehicle Detection**: Utilizes YOLOv10 for efficient and accurate vehicle detection. 🔍
- **Object Tracking**: Employs ByteTrack for robust tracking of vehicles across frames. 📊
- **Data Logging**: Counts and logs the number of vehicles passing through the specified area for analysis. 📈

## Technologies Used

- **OpenCV**: For image processing and video handling. 🖼️
- **YOLOv10**: For state-of-the-art object detection. ⚡
- **ByteTrack**: For online tracking of detected vehicles. 🔄
- **Python**: Programming language used for implementation. 🐍

## Project Structure

```plaintext
.
├── main.py                    # Main script for car counting
├── video/                     # Folder for input videos
│   └── your_input_video.mp4   # Example input video file
├── output_video/              # Folder for output videos
│   └── counted_output.mp4     # Example output video file                 # YOLOv10 model weights and configuration
└── README.md                  # Project documentation
