# Computer-Vision
Computer Vision is a field of Artificial Intelligence that enables machines to interpret and understand visual information from the world, such as images and videos. Human vision, computer vision allows robots and systems to make decisions based on visual inputs.

# YOLOv8 Object Detection - Full Workflow

## 1. LabelImg Installation for Dataset Annotation

### ✅ Steps to Install LabelImg:

**For Windows:**

1. **Install Python (if not installed):**  
   Download and install Python 3.x from [https://www.python.org/downloads/](https://www.python.org/downloads/).

2. **Install LabelImg using pip:**  
   ```bash
   pip install labelImg
   ```

3. **Run LabelImg:**  
   ```bash
   labelImg
   ```

4. **Annotate images** and save labels in YOLO format.

---

## 2. Anaconda Installation and Environment Setup

### ✅ Steps to Install Anaconda:

1. **Download Anaconda:**  
   Visit [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download Anaconda for Windows.
   [https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe]

2. **Install Anaconda:**  
   Run the installer and follow the instructions to complete installation.

3. **Create a new environment for YOLOv8:**  
   ```bash
   conda create -n yolov8 python=3.10
   ```

4. **Activate the environment:**  
   ```bash
   conda activate yolov8
   ```

5. **Install PyTorch (CUDA if GPU available):**  
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

6. **Install YOLOv8 (Ultralytics):**  
   ```bash
   pip install ultralytics
   ```

---

## 3. Dataset Collection and Annotation

- Collect images of the object(s) you want to detect.
- Use **LabelImg** to annotate images and save them in YOLO format.

📂 Example Dataset Structure:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 4. Training YOLOv8 Model

### ✅ Prepare dataset.yaml:
```yaml
path: C:/path/to/dataset
train: images/train
val: images/val
nc: 1  # number of classes
names: ['class_name']  # replace with your class name
```

### ✅ Run training:
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640
```

- **model**: Choose appropriate (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
- **epochs**: Number of training epochs.
- **imgsz**: Image size (default 640).

🖼️ **(Add training progress screenshots here)**

---

## 5. Inference (Prediction)

### ✅ Predict on an image:
```bash
yolo task=detect mode=predict model="C:/Users/Pavan/runs/detect/train16/weights/best.pt" source="C:/Users/Pavan/Desktop/pot.jpg"
```

### ✅ Predict on a video:
```bash
yolo task=detect mode=predict model="C:/Users/Pavan/runs/detect/train16/weights/best.pt" source="C:/Users/Pavan/Desktop/test_video.mp4" show=True save=True
```

- **show=True**: Display output.
- **save=True**: Save output video.

🖼️ **(Add output prediction screenshots here)**

---
## Step 6: Evaluate Model

To validate your model:

```
yolo task=detect mode=val model=runs/detect/trainXX/weights/best.pt data=data.yaml
```

Results like mAP, Precision, and Recall will be displayed and saved.

---
## 7. Export Model to Other Formats

To export model (ONNX, TensorRT, CoreML, etc.):
```bash
yolo export model="C:/Users/Pavan/runs/detect/train16/weights/best.pt" format=onnx
```

- Replace `onnx` with `tflite`, `torchscript`, `coreml`, etc.

---

## 8. Training on Google Colab (Optional)

- Install YOLOv8:
```python
!pip install ultralytics
```

- Mount Google Drive to access dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```

- Use similar training commands as mentioned above.

---

## 9. Additional Notes

- Ensure dataset is properly labeled.
- Use GPU for faster training.
- Monitor training using TensorBoard or Weights & Biases (optional).

---

## 10. References

- Ultralytics YOLOv8: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- LabelImg: [https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)

---


## 📧 Contact

For any queries:

- **Name**: Pavanakumar Walikar
- **Email**: pavan.sw2024@gmail.com
---

---
