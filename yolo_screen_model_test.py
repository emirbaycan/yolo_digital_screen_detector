import argparse
import sys
import os
from ultralytics import YOLO

def run_inference(model_path, input_path, output_path, conf=0.5):
    if not os.path.exists(input_path):
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    results = model(input_path, conf=conf)
    for idx, result in enumerate(results):
        save_path = output_path
        if os.path.isdir(output_path):
            base = os.path.basename(input_path)
            name, ext = os.path.splitext(base)
            save_path = os.path.join(output_path, f"{name}_detected.png")
        result.save(filename=save_path)
        print(f"Saved result to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Screen Detection Console App")

    parser.add_argument('--model', type=str,
                        default="best.pt",
                        help='Path to trained YOLO model .pt file (default: runs/detect/train10/weights/best.pt)')
    parser.add_argument('--input', type=str,
                        default="test_image.png",
                        help='Path to input image (default: camera_images/frame_1738193401.png)')
    parser.add_argument('--output', type=str,
                        default="test_image.png",
                        help='Output file or directory (default: output.png)')
    parser.add_argument('--conf', type=float,
                        default=0.5,
                        help='Confidence threshold (default: 0.5)')

    args = parser.parse_args()
    run_inference(args.model, args.input, args.output, args.conf)
