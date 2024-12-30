## Wildfire Detection Using Divide and Conquer
A Python application for detecting wildfires in satellite imagery using computer vision and divide-and-conquer algorithms.

## Overview
This project implements wildfire detection by analyzing satellite images through:
* Color-based fire and smoke detection in HSV space
* Divide-and-conquer image analysis
* Risk level assessment
* Visual result presentation

## Dataset
Uses the [Satellite Wildfire Detection Dataset](https://universe.roboflow.com/htw-berlin-xv7eo/satellite-wildfire-detection) from Roboflow.

## Requirements
numpy
opencv-python
matplotlib
scipy
pillow

## Installation
# Clone repository
git clone https://github.com/yourusername/Wildfire-Detection-Using-Divide_and_Conquer.git
# Install dependencies
pip install -r requirements.txt

## Usage
1. Run the application:
2. Click "Pilih Gambar" to select an image
3. View detection results and analysis

## Features
* Fire and smoke detection
* Risk level classification (low/medium/high/critical)
* Confidence scoring
* Visual overlays with bounding boxes
* Statistical analysis display

## Technical Implementation
* Divide-and-conquer algorithm for image analysis
* HSV color space segmentation
* Recursive block analysis
* Zone merging for overlapping detections
* Confidence scoring based on color and texture

## Project Structure
app.py          # Main application code
├── FireZone    # Data class for detected zones
├── ForestFireDetector  # Core detection algorithm
└── FireDetectionApp    # GUI interface
