import os
import xml.etree.ElementTree as ET

# Define paths
VOC_ANNOTATIONS_DIR = "screen_dataset/annotations"
YOLO_TRAIN_LABELS_DIR = "datasets/labels/train"
YOLO_VAL_LABELS_DIR = "datasets/labels/val"

# Ensure output directories exist
os.makedirs(YOLO_TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(YOLO_VAL_LABELS_DIR, exist_ok=True)

# Define class mappings
CLASS_DICT = {"screen": 0}  # Assuming only one class

# Get list of train and val image filenames
train_images = set(os.listdir("datasets/images/train"))
val_images = set(os.listdir("datasets/images/val"))

def convert_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get filename and corresponding label file path
    filename = root.find("filename").text.replace(".png", ".txt")
    
    # Check if this image belongs in train or val set
    if filename.replace(".txt", ".png") in train_images:
        yolo_filepath = os.path.join(YOLO_TRAIN_LABELS_DIR, filename)
    elif filename.replace(".txt", ".png") in val_images:
        yolo_filepath = os.path.join(YOLO_VAL_LABELS_DIR, filename)
    else:
        print(f"⚠️ Warning: {filename} not found in train/val sets, skipping!")
        return

    # Get image size
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    with open(yolo_filepath, "w") as yolo_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_DICT:
                continue  # Skip unknown classes

            class_id = CLASS_DICT[class_name]
            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)

            # Convert bounding box to YOLO format (normalized)
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Write to YOLO format
            yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Convert all XML files
for xml_file in os.listdir(VOC_ANNOTATIONS_DIR):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(VOC_ANNOTATIONS_DIR, xml_file))

print("✅ YOLO conversion complete!")