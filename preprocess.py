import os
import librosa
import math
import json
import csv

DATASET_PATH = "fma_small"  # directory where track files are located
METADATA_PATH = "tracks.csv"  # CSV file with track metadata
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, metadata_path, json_path, n_mfcc = 13, n_fft = 512, hop_length = 512, num_segments = 5):

    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_verctors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    genre_dict = {}  # for storing genre label mapping

    with open(metadata_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            track_id, genre = row[0], row[1]  # assuming track_id is first column and genre is second
            file_path = os.path.join(dataset_path, str(track_id) + '.mp3')

            if os.path.exists(file_path):
                if genre not in genre_dict:
                    genre_dict[genre] = len(genre_dict)
                    data["mapping"].append(genre)

                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_verctors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(genre_dict[genre])
                        print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, METADATA_PATH, JSON_PATH, num_segments=10)
