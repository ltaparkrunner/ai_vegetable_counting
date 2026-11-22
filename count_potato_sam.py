import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import SAM


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


# 1. Path to input image (SAM weights will auto-download on first run)
image_path = "potatoes_2.jpg"
if not os.path.exists(image_path):
    print(f"❌ Error: Image '{image_path}' not found!")
    sys.exit(1)

# 2. Verify GPU functionality and load SAM 2.1 Base model
# Options: "sam2.1_t.pt" (tiny/fastest), "sam2.1_s.pt" (small), "sam2.1_b.pt" (base), or "sam_b.pt" (SAM 1)
compute_device = get_compute_device()
model = SAM("sam2.1_b.pt")


def count_potatoes_sam(img_path, min_area_ratio=0.001, max_area_ratio=0.08):
    """
    Segment all objects using SAM and filter masks by relative area
    to exclude background and tiny noise fragments.
    """
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Could not read the image file.")
        return

    img_h, img_w = img.shape[:2]
    total_pixels = img_h * img_w
    min_area = total_pixels * min_area_ratio
    max_area = total_pixels * max_area_ratio

    # 3. Run zero-shot automatic mask generation on the verified device
    print("🚀 Running Segment Anything (SAM) inference...")
    results = model.predict(
        source=img_path,
        device=compute_device,
        conf=0.40,
        iou=0.50
    )
    res = results[0]  # Extract the result object for the first image

    # 4. Filter detected masks by area and draw contours
    valid_potatoes = 0
    if res.masks is not None:
        polygons = res.masks.xy
        for poly in polygons:
            if len(poly) < 3:
                continue

            points = np.array(poly, dtype=np.int32)
            area = cv2.contourArea(points)

            # Skip masks that are too small (dirt/noise) or too large (entire pile/background)
            if area < min_area or area > max_area:
                continue

            valid_potatoes += 1

            # Generate a random bright BGR color for each contour
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))
            )
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        print(f"\n🎉 COUNT RESULT (SAM): Found {valid_potatoes} potato(es)!")
    else:
        print("\n❌ SAM did not detect any masks.")

    # 5. Save the annotated output image
    output_path = "counted_result_sam.jpg"
    cv2.imwrite(output_path, img)
    print(f"💾 Annotated result saved to '{output_path}'")


if __name__ == "__main__":
    count_potatoes_sam(image_path)
