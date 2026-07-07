# Vision_Inspection_Pipeline

An automated multi-camera image processing and inference system designed for real-time manufacturing monitoring.

## Overview
This system provides an end-to-end pipeline for industrial production lines. It monitors designated directories for multi-angle camera images (Left, Center, Right), processes them through a YOLO-based inference model, and automatically logs production counts and inspection results into a MariaDB database.

## Key Features
* **Real-time Monitoring:** Uses watchdog observers to detect and process incoming image files instantly.
* **Multi-Camera Synchronization:** Automatically groups images from multiple camera angles (Left, Center, Right) based on file timestamps[cite: 4].
* **AI-Powered Inference:** Integrates Ultralytics YOLO models to perform object detection and classification on production line items[cite: 4].
* **Database Integration:** Manages production data, including image metadata and product counts, via MariaDB[cite: 1, 4].
* **Automated Logging:** Tracks production quantities per product type and monitors for detection errors[cite: 4].

## Workflow
1. **Capture:** Cameras save images to monitored network folders[cite: 4].
2. **Dispatch:** The system detects files and buffers them until a full set (Left, Center, Right) is ready[cite: 4].
3. **Inference:** The YOLO model runs a prediction to identify objects[cite: 4].
4. **Processing & Storage:** Results are saved, merged, and uploaded to the database[cite: 4].

## Technical Stack
* **Core:** Python, OpenCV, Ultralytics YOLO[cite: 4]
* **Database:** MariaDB
* **File Management:** Pathlib, Watchdog[cite: 2, 4]

## Usage
1. Configure database connection settings in `db_settings.json`[cite: 1].
2. Define directory paths in `paths.py`.
3. Run the pipeline:
   ```bash
   python main.py