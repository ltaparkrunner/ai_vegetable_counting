import os
import sys
import re
from pathlib import Path

directory_path = "potato_piles"
if not os.path.exists(directory_path):
    print(f"❌ Error: Directory '{directory_path}' not found!")
    sys.exit(1)

folder = Path(directory_path)

for file_path in sorted(folder.iterdir()):
    if not file_path.is_file():
        continue

    # Match the first '_jpg', '.jpg', '_jpeg', or '.jpeg' (case-insensitive)
    match = re.search(r'[._](jpe?g)', file_path.name, flags=re.IGNORECASE)
    if not match:
        continue

    # Replace '_' with '.' so 'img10_jpeg...' becomes 'img10.jpeg'
    base_name = file_path.name[:match.start()]
    ext = match.group(1).lower()
    new_name = f"{base_name}.{ext}"

    # Skip if the filename is already clean
    if new_name == file_path.name:
        continue

    new_path = file_path.with_name(new_name)

    # Skip if a file with the new name already exists
    if new_path.exists():
        print(f"⏭️ Skipped (already exists): {file_path.name} -> {new_name}")
        continue

    file_path.rename(new_path)
    print(f"✅ Renamed: {file_path.name} -> {new_name}")
