from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Uses your existing trained YOLOv8n-seg model on AMD GPU
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/segment/train-2/weights/best.pt",
    confidence_threshold=0.25,
    device="cuda:0"
)

result = get_sliced_prediction(
    "potatoes_2.jpg",
    detection_model,
    slice_height=256,          # Small tiles zoom in on tightly packed potatoes
    slice_width=256,
    overlap_height_ratio=0.30, # 30% overlap prevents cutting edge potatoes in half
    overlap_width_ratio=0.30,
    postprocess_type="NMM",    # Non-Maximum Merging combines split edge detections
    postprocess_match_threshold=0.45
)

print(f"🥔 SAHI Count: {len(result.object_prediction_list)}")
result.export_visuals(export_dir="sahi_out/", hide_labels=True, hide_conf=True)
