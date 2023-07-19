import os
import librosa
import math
import json

INPUT_PATH = "Input"
JSON_PATH = "dataInput.json"
SAMPLE_RATE = 22050
DURATION = 30  # total duration of each audio file, in seconds
NUM_SEGMENTS = 10  # change this to match your training code

def save_mfcc(input_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    # dictionary to store data
    data = {
        "mfcc": []  # (inputs)mfcc vectors for each of the labels
    }

    num_samples_per_segment = int(SAMPLE_RATE * DURATION / NUM_SEGMENTS)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for file in os.listdir(input_path):
        # ensure we're processing .wav files only
        if not file.endswith('.wav'):
            continue

        # load audio file
        file_path = os.path.join(input_path, file)
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # process segments extracting mfcc and storing data
        for s in range(NUM_SEGMENTS):
            start_sample = num_samples_per_segment * s
            finish_sample = start_sample + num_samples_per_segment

            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length)
            mfcc = mfcc.T

            # store MFCC for segment if it has the expected length
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                print("{}, segment:{}".format(file_path, s+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(INPUT_PATH, JSON_PATH)
