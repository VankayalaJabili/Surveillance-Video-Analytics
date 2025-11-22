# 📹 Surveillance Video Analytics – Anomaly Detection  
AI-based CCTV Footage Monitoring System

## 📖 Overview
Manual surveillance is inefficient and prone to human error. This project implements an AI-powered anomaly detection system using **Computer Vision** and **Deep Learning** to automatically identify unusual or suspicious activities in CCTV video feeds.

The system detects:
- Unauthorized entry  
- Sudden abnormal movements  
- Crowd violence  
- Suspicious objects or events  
- Any deviation from normal behavior  


## 🎯 Research Goal
To build an AI-based anomaly detection system that analyzes video feeds and automatically identifies unusual activities in real-time, reducing human dependency and improving surveillance accuracy.

## ✅ Objectives
- Detect abnormal activities in CCTV footage  
- Reduce reliance on manual monitoring  
- Provide real-time alerts  
- Improve overall surveillance accuracy  
- Use CNN-based models for spatial–temporal analysis  

## 📁 Dataset  
Datasets used: UCSD Anomaly Detection Dataset. This contain both normal and abnormal video activities required for training.

## 📦 Technologies & Libraries
- **Python**
- **OpenCV** – Video processing  
- **TensorFlow / Keras** – Deep learning  
- **NumPy, Pandas** – Data analysis  
- **Matplotlib** – Visualization  
- **Scikit-learn** – Model evaluation  
- Install required packages:
     pip install opencv-python tensorflow keras numpy pandas matplotlib scikit-learn

## 🧪 Project Workflow  
### 1. Data Retrieval  
  Loaded all CCTV videos using OpenCV’s VideoCapture and extracted frames from each video.

### 2. Data Preparation  
• Extracted video frames  
• Resized frames to 128×128  
• Normalized pixel values  
• Encoded labels (0 = normal, 1 = anomaly)  
• Performed train-test split using 80-20 ratio  

### 3. Data Visualization  
• Used optical flow to analyze motion  
• Applied frame differencing  
• Visualized motion intensity graphs  
Anomalies appeared as sharp spikes.

### 4. Feature Selection  
• Selected extracted video frames as input features.  
• Labels were used to distinguish normal and anomalous activities.

### 5. Model Building  
Used a CNN (Convolutional Neural Network) to learn spatial features from frames.  
• Conv2D + MaxPooling layers  
• Flatten + Dense layers  
• Sigmoid output for binary classification  

### 6. Model Evaluation  
Evaluated using:  
• Accuracy  
• Precision  
• Recall  
• F1-score  
• Reconstruction error threshold  
Frames above threshold were flagged as anomalous.

### 7. Real-Time Prediction  
The model detected abnormal frames and visualized the difference between normal and anomalous events using bounding boxes and output graphs.

## 📊 Output Summary  
• Model showed good performance in identifying anomalies  
• Motion graphs clearly revealed unusual activity  
• CNN effectively extracted spatial features from frames  
