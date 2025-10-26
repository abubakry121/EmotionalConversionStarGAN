import os
import argparse
import numpy as np
from utils.data_preprocessing_utils import get_wav_and_labels

# Emotion mapping
EMOTION_LABELS = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgust": 5
}

def save_processed(data_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for sample in data_list:
        file_name = os.path.basename(sample["path"]).replace(".wav", ".npz")
        out_path = os.path.join(out_dir, file_name)
        np.savez(
            out_path,
            mel=sample["mel"],
            f0=sample["f0"],
            energy=sample["energy"],
            label=sample["label"]
        )

def main(data_path, out_dir):
    print(f"Processing dataset in: {data_path}")
    data_list = get_wav_and_labels(data_path, EMOTION_LABELS)
    print(f"Found {len(data_list)} audio files")
    save_processed(data_list, out_dir)
    print(f"Processed features saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder containing emotion subfolders")
    parser.add_argument("--out_dir", type=str, default="./processed_data",
                        help="Folder to save processed features (.npz)")
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
