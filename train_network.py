# USAGE
# python train_network.py --dataset dataset --model fashion.model --labelbin mlb.pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
args = vars(ap.parse_args())

# initialize hyperparameters - the number of epochs, initial learning rate, batch size, and image dimensions
EPOCHS = 150
INIT_LR = 1e-3
BS = 64
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] Loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # extract set of class labels from the image path and update the labels list
    label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels using scikit-learn's special multi-label binarizer implementation
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# increase the training set size using data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# initialize the model and use sigmoid activation function in the final layer of the network
print("[INFO] Compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
                            final_act="sigmoid")

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] Training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] Serializing network...")
model.save(args["model"])

# save the multi-label binarizer to disk
print("[INFO] Serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training/Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("Loss plot.png")

# plot the training accuracy
plt.clf()
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training/Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("Accuracy plot.png")
