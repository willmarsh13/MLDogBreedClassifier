#####################################
# Author: William Marsh             #
# Date: 5/06/2025                   #
# Class: ECE 5494                   #
#####################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from tqdm import tqdm
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load the dataset
def load_dataset(path):
    data = load_files(path)
    file_paths = np.array(data['filenames'])
    targets = np.array(data['target'])
    class_names = np.array(data['target_names'])
    return file_paths, targets, class_names

test_files, test_targets, class_names = load_dataset('./dogImages/test')

# convert to tensors using the same function from other files.
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    return np.vstack([path_to_tensor(img_path) for img_path in tqdm(img_paths)])

test_tensors = preprocess_input(paths_to_tensor(test_files))

# load trained model from current directory
# TODO: you must have run the model previously for this to work.
from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

xception_bottleneck = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in xception_bottleneck.layers:
    layer.trainable = False

model = Sequential([
    xception_bottleneck,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(133, activation='softmax')
])

model.load_weights('saved_model_Xception.h5')

# predictions
predictions = model.predict(test_tensors, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)

# calculate the accuracy for each breed
num_classes = len(class_names)
correct_per_class = np.zeros(num_classes)
total_per_class = np.zeros(num_classes)

for true_label, pred_label in zip(test_targets, predicted_classes):
    total_per_class[true_label] += 1
    if true_label == pred_label:
        correct_per_class[true_label] += 1

accuracy_per_class = correct_per_class / total_per_class

# prints the accuracy by class
plt.figure(figsize=(18, 6))
plt.bar(range(num_classes), accuracy_per_class, color='skyblue')
plt.xlabel('Dog Breed (class index)')
plt.ylabel('Accuracy')
plt.title('Per-Class (Breed) Accuracy for Xception Model')
plt.xticks(range(num_classes), [name.split('.')[-1] for name in class_names], rotation=90, fontsize=6)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('per_class_accuracy_xception.png', dpi=300)

plt.show()