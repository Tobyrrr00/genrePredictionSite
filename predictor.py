import json
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("genrePredictionSite\genreClassifier150")

# Load data from json file
with open("dataInput.json", "r") as fp:
    data = json.load(fp)

# We convert lists into numpy arrays
inputs = np.array(data["mfcc"])

# Ensure that input data is of the same shape as the model's input layer
# inputs = inputs[..., np.newaxis]

# Make predictions
predictions = model.predict(inputs) # this will return probabilities for each genre
predicted_indexes = np.argmax(predictions, axis=1) # this will return the index of the genre with highest probability

# If you have the genre labels stored (for example in a list, where each index corresponds to a genre), you can use the indexes to get the genre names
genre_labels = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock", "Classical", "Country", "Jazz", "Old-Time / Historic", "Soul-RnB", "Spoken", "Blues", "Easy Listening"]
predicted_genres = [genre_labels[i] for i in predicted_indexes]

print(predicted_genres)

