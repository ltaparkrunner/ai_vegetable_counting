import os
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


def start_training():
    # Select the best available functional device
    compute_device = get_compute_device()

    # Use 'yolov8n-seg.pt' or 'yolov8s-seg.pt' for integrated GPUs / CPUs
    model = YOLO("yolov8n-seg.pt")

    yaml_paths = ("ai_train_potatoes/data.yaml", "data.yaml")
    yaml_path = next((p for p in yaml_paths if os.path.exists(p)), None)

    if not yaml_path:
        print(f"❌ Error: None of the config files {yaml_paths} were found!")
        return

    print(f"🚀 Starting training using '{yaml_path}'...")

    # Adjust batch size and AMP automatically based on device
    use_gpu = compute_device == 0

    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=4 if use_gpu else 8,
        device=compute_device,
        amp=False,        # Keep FP32 to prevent MIOpen FP16 crashes on AMD gfx1103
        workers=2,
        mosaic=1.0
    )

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    start_training()
