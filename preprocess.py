import os
import librosa
import math
import json
import csv
import numpy as np

DATASET_PATH = "fma_small"  # directory where track files are located
METADATA_PATH = "fma_metadata/fma_small_genres.csv"  # CSV file with track metadata
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, metadata_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Use numpy to load the CSV metadata
    metadata = np.genfromtxt(metadata_path, delimiter=',', dtype=None, encoding=None, names=True)
    track_genre_dict = {row['track_id']: row['genre_top'] for row in metadata}

    unique_genres = list(set(track_genre_dict.values()))
    genre_to_label = {genre: i for i, genre in enumerate(unique_genres)}
    data["mapping"] = unique_genres

    for track_id, genre in track_genre_dict.items():
        file_path = os.path.join(dataset_path, str(track_id).zfill(6)[:3], f"{str(track_id).zfill(6)}.mp3")

        if os.path.exists(file_path):
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                mfcc = mfcc.T

                if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(genre_to_label[genre])
                    print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, METADATA_PATH, JSON_PATH, num_segments=10)
