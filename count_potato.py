import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_compute_device():
    """Check if a GPU exists and can execute a real Conv2d operation."""
    if not torch.cuda.is_available():
        print("⚙️ No GPU detected. Falling back to CPU.")
        return "cpu"

    try:
        device_name = torch.cuda.get_device_name(0)
        # Run a quick FP32 convolution test to verify ROCm/CUDA kernels work
        x = torch.randn(1, 3, 32, 32, device="cuda")
        conv = torch.nn.Conv2d(3, 8, kernel_size=3).to("cuda")
        _ = conv(x)
        torch.cuda.synchronize()
        print(f"⚙️ GPU verified and functional: {device_name} (device=0)")
        return 0
    except Exception as e:
        print(f"⚠️ GPU detected ({torch.cuda.get_device_name(0)}), but functional test failed: {e}")
        print("⚙️ Falling back to CPU.")
        return "cpu"


# 1. Path to the trained model weights and input image
model_path = "runs/segment/train-2/weights/best.pt"
image_path = "potatoes_2.jpg"

if not os.path.exists(model_path) or not os.path.exists(image_path):
    print(f"❌ Error: Check your file paths!\nModel: {model_path}\nImage: {image_path}")
    sys.exit(1)

# 2. Verify GPU functionality and load the model once
compute_device = get_compute_device()
model = YOLO(model_path)


def count_potatoes(img_path):
    # 3. Run inference on the verified device (FP32 is used by default)
    results = model.predict(
        source=img_path,
        conf=0.25,
        iou=0.40,
        device=compute_device
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
