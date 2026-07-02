import os
import subprocess

data_dir = r"E:\iotj_revision\data"

for device in os.listdir(data_dir):
    device_path = os.path.join(data_dir, device)
    if not os.path.isdir(device_path):
        continue
    for f in os.listdir(device_path):
        if f.endswith(".rar"):
            rar_path = os.path.join(device_path, f)
            print(f"Extracting: {rar_path}")
            # Uses 7-Zip command line
            subprocess.run([
                r"C:\Program Files\7-Zip\7z.exe",
                "e", rar_path,
                f"-o{device_path}",
                "-y"
            ])

print("\nAll done!")