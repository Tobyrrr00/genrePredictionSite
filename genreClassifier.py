#This is a multiclass Classifier which Classifies given music into different genres
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

dataset_path = "data.json"

def load_data(dataset_path):

    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    
    #convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    print("Data succesfully loaded!")

    return inputs, targets

def plot_history(history):

    fog, axs = plt.subplots(2)

    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc = "lower right")
    axs[0].set_title("Accuracy eval")

    #create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc = "lower right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    #load data
    inputs, targets = load_data(dataset_path)

    #split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, 
                                                                                targets,
                                                                                test_size = 0.3)
    
    #build the network architecture
    model = tf.keras.Sequential([
        #input layer
        #flatten input layer. 
        tf.keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        #1st hidden layer
        #We are using ReLU activation function instead of sigmoid.
        #for ReLU(h) = {0 if h<0, h if h>=0}
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(0.3), #Using Dropout and Regularizers to control Overfitting 

        #2nd hidden layer
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(0.3),

        #3rd hidden layer
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(.001)),
        tf.keras.layers.Dropout(0.3),

        #4th hidden layer
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    #compile the network
    opimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    model.summary()

    #train network

    history = model.fit(inputs_train, targets_train, 
             validation_data = (inputs_test, targets_test),
             epochs = 150,
             batch_size = 32)
    
    #plot accuracy and error over the epochs
    plot_history(history)

    #Save the mode
    model.save("genreClassifier150")