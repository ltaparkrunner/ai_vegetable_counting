import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Path to the trained model weights and input image
model_path = "runs/segment/train-2/weights/best.pt"
image_path = "potatoes_2.jpg"

if not os.path.exists(model_path) or not os.path.exists(image_path):
    print(f"❌ Error: Check your file paths!\nModel: {model_path}\nImage: {image_path}")
    sys.exit(1)

# 2. Automatically select GPU (device=0) if ROCm/CUDA is available, otherwise fall back to CPU
device = 0 if torch.cuda.is_available() else "cpu"
print(f"⚙️ Active device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if device == 0 else 'CPU'}")

# Load the model once outside the function to avoid reloading weights into GPU VRAM on every image
model = YOLO(model_path)


def count_potatoes(img_path):
    # 3. Run inference on the GPU (FP32 is used by default in predict)
    results = model.predict(
        source=img_path,
        conf=0.25,
        iou=0.40,
        device=device
    )
    res = results[0]  # Extract the result object for the first image

    # 4. Load the clean original image to draw polygon outlines
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Could not read the image file.")
        return

    # 5. Check for detected segmentation masks and draw their contours
    if res.masks is not None:
        potato_count = len(res.masks)
        print(f"\n🎉 COUNT RESULT: Found {potato_count} potato(es)!")

        # res.masks.xy contains a list of arrays, each holding (x, y) pixel coordinates for one potato
        polygons = res.masks.xy
        for poly in polygons:
            if len(poly) == 0:
                continue

            # Convert polygon coordinates to integers for OpenCV
            points = np.array(poly, dtype=np.int32)

            # Generate a random bright BGR color for each contour so adjacent potatoes stand out
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))
            )

            # Draw the closed polygon outline (thickness=2 sets line width in pixels)
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
    else:
        print("\n❌ No objects detected. Try lowering the 'conf' threshold.")

    # 6. Save the annotated output image
    output_path = "counted_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"💾 Annotated result saved to '{output_path}'")


if __name__ == "__main__":
    count_potatoes(image_path)
