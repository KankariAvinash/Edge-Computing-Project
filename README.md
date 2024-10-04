# Real-Time Predictive Maintenance for Industrial IoT Using Edge Intelligence
Machinery downtime in industrial settings can lead to significant productivity losses and revenue reduction. Un-planned maintenance and sudden equipment failures pose major operational challenges. This project focuses on developing predictive maintenance system using edge computing to monitor machine health, forecast maintenance needs, and mitigate unexpected failures.

The system is deployed on edge computing boards, integrated with temperature and acoustic sensors for comprehensive machinery condition monitoring. By leveraging edge computing and advanced machine learning algorithms, this system aims to minimize unplanned downtime, enhancing operational efficiency in industrial environments.

## Methodology
The proposed methodology integrates edge intelligence with predictive maintenance strategies in Industrial IoT (IIoT) environments, utilizing deep learning models on resource constrained devices. The core of the system is built around the ESP32 microcontroller, chosen for its low power consumption and capability to support real-time data processing. The ESP32 is leveraged to perform on-device inference using deep learning techniques, enabling real-time detection of anomalies in industrial equipment. The methodology is designed to address the unique challenges of IIoT systems, where network latency, bandwidth limitations, and real-time constraints must be managed efficiently. The deep learning model is trained to classify normal and anomalous behaviors of industrial machinery, using audio signals as input. These signals are processed locally on the ESP32, which ensures timely prediction and minimizes re-liance on cloud-based infrastructure. In this section, we detail the processes involved in data collection, model development, deployment on the ESP32, and integration with edge intelligence techniques to achieve real-time predictive maintenance.As previously mentioned, the ESP32 was selected as the device for executing the model. Specifically, the ESP32-S3-EYE variant was chosen due to its integrated digital microphone. Additionally, it is equipped with 8 MB of Octal PSRAM and 8 MB of flash memory, which provides sufficient resources for real-time neural network execution. The device also benefits from support provided by the ESP-IDF libraries, real-time operating system (RTOS) capabilities, and compatibility with the Edge Impulse platform, facilitating the deployment of the model on the device.
![image](https://github.com/user-attachments/assets/cabeef9d-1d3f-462f-a60e-9023d7dbd4f4)

## Results
The anomaly detection algorithm, Audio signals are captured from the microphone and classified by a deep learning model into two categories: normal behavior (high speed) and abnormal behavior (low speed). In conjunction with temperature readings, a decision-making algorithm further categorizes the signals into four conditions:
- normal behavior (Green): high speed and temperature below threshold (T max).
- abnormal temperature (Yellow): high speed and temperature above threshold (T max).
- abnormal speed (Blue): low speed and temperature below threshold (T max).
- abnormal behavior (Red): low speed and temperature above threshold (T max).
  
![image](https://github.com/user-attachments/assets/76bbeea8-e69e-4ce2-a463-c82b83717108)
