# Vision_Inspection_Pipeline

An automated multi-camera image processing and inference system designed for real-time manufacturing monitoring.

## Overview

This project implements a real-time vision inference pipeline that integrates Watchdog-based filesystem monitoring with a YOLO object detection model.

The system collects images from multiple camera views (Left, Center, and Right), synchronizes image sequences using timestamp-based buffering, performs AI inference, and stores the processed results into a relational database.

The pipeline is designed with a multi-threaded architecture to separate image monitoring and inference processing, enabling stable real-time data handling.

---

## Key Features

- **Real-Time Image Monitoring**
  - Detects newly created image files using the Watchdog filesystem event handler.
  - Filters valid image files and sends events to a processing queue.

- **Multi-Camera Synchronization**
  - Collects images from multiple camera views (Left, Center, Right).
  - Uses timestamp-based buffering to synchronize images into a complete inference set.

- **Multi-View AI Inference Support**
  - Designed to support custom-trained deep learning models that process synchronized images from multiple camera views simultaneously.
  - Enables multi-view classification and analysis using combined Left, Center, and Right image inputs.
 
- **Multi-Threaded Processing Pipeline**
  - Separates file monitoring and AI inference into independent threads.
  - Uses Queue and Buffer mechanisms for thread-safe data exchange.

- **YOLO-Based Object Detection**
  - Integrates Ultralytics YOLO models for object detection.
  - Processes synchronized multi-view images and aggregates inference results.

- **Database Integration**
  - Stores inference results, image metadata, and processing logs in MariaDB.
  - Provides structured data management for monitoring and analysis.
 
---

## Workflow

### 1. Image Capture & Monitoring

- Images generated from external sources or camera systems are stored in monitored directories.
- Watchdog detects file creation events and transfers image paths to the processing queue.

### 2. Buffering & Image Synchronization

- Dispatcher receives image events from the queue.
- Images are temporarily stored in buffers based on camera view and timestamp.
- When all required camera images are collected, an inference task is created.

### 3. AI Inference Processing

- Processor receives synchronized image sets from the buffer.
- The collected Left, Center, and Right images are provided as a unified input for the inference model.
- Detection results are combined and processed according to the application logic.

### 4. Result Storage

- Processed inference results and metadata are stored in MariaDB.
- Database records are used for monitoring and further analysis.

---

## Technical Stack

- **Language**
  - Python 3.x

- **Computer Vision / AI**
  - Ultralytics YOLO
  - OpenCV

- **Event Monitoring**
  - Watchdog

- **Concurrency**
  - Threading
  - Queue

- **Database**
  - MariaDB
  - PyMySQL / MariaDB Connector

---

## Usage

### 1. Install Dependencies

```bash
pip install ultralytics watchdog mariadb opencv-python