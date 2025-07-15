
 #####################################
 # Author: William Marsh             #
 # Date: 4/02/2025                   #
 # Class: ECE 5494                   #
 #####################################

import numpy as np
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load_dataset
#    - reads subdirectories and stores it as label
#    - ex: if the folder is golden_retriever, that's the label
def load_dataset(path):
    data = load_files(path)
    file_paths = np.array(data['filenames'])
    targets = to_categorical(np.array(data['target']), num_classes=133)
    return file_paths, targets

# split data into training files and targets for train, validation, and test.
train_files, train_targets = load_dataset('./dogImages/train')
valid_files, valid_targets = load_dataset('./dogImages/valid')
test_files, test_targets = load_dataset('./dogImages/test')

# path_to_tensor
#    - converts image to a 4d tensor (required by Keras)
#    - size is (1, 244, 244, 3)
def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

# paths_to_tensor
#    - loops through every image path, stacks into one tensor
def paths_to_tensor(img_paths):
    return np.vstack([path_to_tensor(img_path) for img_path in tqdm(img_paths)])

# normalize pixel values
train_tensors = preprocess_input(paths_to_tensor(train_files))
valid_tensors = preprocess_input(paths_to_tensor(valid_files))
test_tensors  = preprocess_input(paths_to_tensor(test_files))

# Load pretrained ResNet50
resnet_bottleneck = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze weights to not update during training to avoid overfitting
for layer in resnet_bottleneck.layers:
    layer.trainable = False

# Add classification layers and prevent overfitting
model = Sequential([
    resnet_bottleneck,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(133, activation='softmax')
])

# Use adam optimizer. Track accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best model (based on loss)
checkpointer = ModelCheckpoint(filepath='saved_model_ResNet50.h5', verbose=1, save_best_only=True)

#Train for 10 epochs with a batch size of 32.
model.fit(
    train_tensors, train_targets,
    validation_data=(valid_tensors, valid_targets),
    epochs=10, batch_size=32, callbacks=[checkpointer]
)

# save best model to hard drive and print out accuracy
model.load_weights('saved_model_ResNet50.h5')
loss, accuracy = model.evaluate(test_tensors, test_targets)
print(f'Test accuracy: {accuracy * 100:.2f}%')
