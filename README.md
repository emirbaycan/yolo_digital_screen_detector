# yolo_digital_screen_detector

**yolo_digital_screen_detector** is an open-source Python tool for detecting and localizing digital screens (monitors, TVs, laptops, tablets, smartphones) in images using YOLO-based deep learning. The project provides fast, accurate detection and can automatically crop detected screen regions for further analysis, privacy masking, or OCR.

---

## Features

- **YOLO-based detection**: Fast, robust, and accurate screen detection using state-of-the-art object detection models.
- **Easy CLI interface**: Run detection on images with a single command.
- **Automatic cropping**: Saves output images with detected screens cropped or highlighted.
- **Flexible input/output**: Works with individual files or output directories.
- **Production-ready**: Minimal setup with requirements.txt and default config values.
- **Ready for video support**: Extendable for video and webcam detection.

---

## Installation

Clone this repository and install all dependencies:

```bash
git clone https://github.com/emirbaycan/yolo_digital_screen_detector.git
cd yolo_digital_screen_detector
pip install -r requirements.txt
```

---

## Usage

**Basic usage with default values:**

```bash
python yolo_screen_model_test.py
```
- Runs detection using the default model and test image.

**Custom usage:**

```bash
python yolo_screen_model_test.py \
  --model path/to/your_model.pt \
  --input path/to/your_image.png \
  --output results/ \
  --conf 0.6
```

**Arguments:**
- `--model`: Path to the trained YOLO model `.pt` file (default: `best.pt`)
- `--input`: Input image file (default: `test_image.png`)
- `--output`: Output file or directory (default: `output.png`)
- `--conf`: Detection confidence threshold (default: `0.5`)

**Example:**  
Detect and save output to a directory:
```bash
python yolo_screen_model_test.py --input my_image.jpg --output ./results/
```

---

## Example

**Input image:**

![Input Example](test_image.png)

**Detection result:**

![Detection Example](tested_image.png)

---

## Model Training

To train your own digital screen detection model:

1. **Annotate your images**  
   - Use [LabelImg](https://github.com/heartexlabs/labelImg) (open-source, GUI-based) to draw bounding boxes around digital screens in your dataset.
   - Save labels in **YOLO format** (`.txt` files with each image).
   - Alternative tools: [makesense.ai](https://www.makesense.ai/), [Roboflow](https://roboflow.com/).

2. **Organize your dataset**
   - Images and labels should be in folders (e.g. `images/`, `labels/`).
   - Each label file should have the same name as its image, but with `.txt` extension.

3. **Train with Ultralytics YOLO**  
   - Follow [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/datasets/detect/) for custom object detection training.
   - Example command:
     ```bash
     yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
     ```

---

## Requirements

All required packages are listed in `requirements.txt`:

```
ultralytics>=8.0.0
torch>=1.8.0
opencv-python>=4.5.0
PyYAML>=5.4.0
av>=10.0.0
matplotlib>=3.6.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Annotation

**Bounding box annotation** for screens was performed using [LabelImg](https://github.com/heartexlabs/labelImg).  
- To label your own dataset, install LabelImg, draw boxes around screens, and export in YOLO format.
- Example workflow:
  1. Install LabelImg: `pip install labelImg`
  2. Run: `labelImg`
  3. Load images, annotate, and save labels as YOLO `.txt` files.

---

## Contributing

Contributions, bug reports, and feature requests are welcome!  
- Fork this repo, make your changes, and open a pull request.
- For suggestions or questions, open an issue.

---

## License

MIT License

---

## Contact

Developed by Emir Baycan  
[GitHub](https://github.com/emirbaycan)

For questions, open an issue or contact via GitHub.

---
