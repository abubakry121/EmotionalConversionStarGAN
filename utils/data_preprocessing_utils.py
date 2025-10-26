import os
from utils.audio_utils import preprocess_wav

def get_wav_and_labels(data_dir, label_dict):
    """
    data_dir: root folder containing emotion subfolders
    label_dict: { emotion_name : id }
    Returns list of dicts: {'path', 'mel', 'f0', 'energy', 'label'}
    """
    data_list = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = os.path.basename(root).lower()
                if emotion not in label_dict:
                    continue

                features = preprocess_wav(file_path)
                data_list.append({
                    "path": file_path,
                    "mel": features["mel"],
                    "f0": features["f0"],
                    "energy": features["energy"],
                    "label": label_dict[emotion]
                })

    return data_list
