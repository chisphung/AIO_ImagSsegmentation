# Self-driving Car Simulator

This project is a modification tailored to the requirements of the UIT Car Racing contest (e.g., fine-tuning, model changes). The edited code is primarily based on the [Self-driving Car Simulator](https://github.com/bmd1905/Self-driving-Car-Simulator).

## Introduction

### Pipeline
<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236652354-843e9a41-3289-435c-be5a-fee681d38f2f.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="400" />
</p>

### Detection Task
We compared different models based on the number of parameters:

<div align="center">

| Model     | Params (M) | mAP@.5 | mAP@.5:.95 |
|-----------|------------|--------|------------|
| YOLOv11n  | 21.2       | 0.993  | 0.861      |
| YOLOv8n   | **3.2**    | 0.992  | **0.887**  |

Fine-tuning for model training: We leveraged Roboflow processing to generate cases with lighting changes and noise. The flip feature was avoided due to incorrect labels in flipped images.

```python
model.train(data="../trafficsign-2-1/data.yaml", epochs=300, imgsz=(640, 384), fliplr=0.0, flipud=0.0)
```

</div>

Based on our results, YOLOv8 Nano was adopted for this project. Below are some example images predicted using YOLOv8 Nano. The model successfully predicted small objects, which is crucial for the control task.

### Datasets
- [Traffic Sign Detection](https://app.roboflow.com/chisphung/trafficsigndetection-7tkoh/1)  
- [Road Segmentation](https://app.roboflow.com/chisphung/roadsegmentation-wlflg/5)  

### Segmentation Task
We initially used YOLOv8m, which met our requirements.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0c476cdd-ab65-4cb0-ac2a-0a29c25e1586" width="700" height="300" />
</p>

### Controller
A PID controller (Proportional Integral Derivative controller) is a widely used control loop mechanism. It continuously calculates an error value as the difference between a desired setpoint and a measured process variable, applying corrections based on proportional, integral, and derivative terms (denoted P, I, and D). The controller minimizes the error over time by adjusting a control variable using a weighted sum of the control terms.

<p align="center">
  <img src="https://user-images.githubusercontent.com/90423581/236684583-6f31d6ff-80eb-44c4-99ee-0df2c42a4f10.png" width="500" height="350" />
</p>

#### Pseudocode for Control

```python
# Find A and B
C = (A + B) * 2.5 / 3.5  # Right center point
error = IC = HI - HC
angle = PID(error, p, i, d)
speed = f(angle)

# OH: Hyper-parameter (e.g., 1/3 or 1/4 of image height)
```

#### Pseudocode for Handling Traffic Signs

```python
# Pre-turning
for detected_class in yolo_outputs:
    if detected_class in traffic_signs:
        majority_classes.add(detected_class)

        if len(majority_classes) == 10:
            majority_class = find_majority(majority_classes)
            turning = True
```

```python
# Turning
while turning_counter <= max_turning_counter:
    switch case for majority_class:
        angle = constant
        speed = constant
        
        turning_counter += 1
```

## Usage
1. Download weights and place them in the `pretrain` directory.
2. Start Unity.
3. Run the script:

```bash
python main.py
```

## Demo
Video demo available: Delulu team [here](https://www.youtube.com/live/oFe0Hyr4-9o?si=h9vrzMw1fAr1PedO).

## Result
- Top 3 in the qualifiers round.
- Could not progress to the final round due to FPS discrepancies causing incorrect fine-tuned parameters.

## References
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [PIDNet](https://github.com/XuJiacong/PIDNet)
- [CEEC](https://github.com/user-attachments/assets/0c476cdd-ab65-4cb0-ac2a-0a29c25e1586)

# CarRacing-ver1
