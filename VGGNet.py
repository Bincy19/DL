#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow import data as tf_data
from sklearn.metrics import recall_score

location = "/home/anbi23bd/DRUMSTER/DRUMSTER"
num_skipped = 0

# Remove corrupted images
for folder_name in ("DAMAGED", "UNDAMAGED"):
    folder_path = os.path.join(location, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

image_size = (180, 180)
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    location,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    location,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Take a subset of the dataset
subset_size = 100
train_subset = train_ds.take(subset_size)
val_subset = val_ds.take(subset_size)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images, training=False):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply data augmentation to the training images.
train_subset = train_subset.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_subset = train_subset.prefetch(tf_data.AUTOTUNE)
val_subset = val_subset.prefetch(tf_data.AUTOTUNE)

# Define the VGG16 model
def make_vgg16_model(input_shape, num_classes):
    base_model = applications.VGG16(
        weights='imagenet', 
        input_shape=input_shape, 
        include_top=False
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model

model_vgg16 = make_vgg16_model(input_shape=image_size + (3,), num_classes=1)

print('Training DAMAGED vs UNDAMAGED dataset using VGG16 - \n')
print("Learning Rate: 0.0001 , Batch Size=64, Epoch=25")

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("vgg16_save_at_{epoch}.keras"),
]

model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

history = model_vgg16.fit(
    train_subset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_subset,
)

print("The results for DAMAGED vs UNDAMAGED Dataset using VGG16")
val_loss, val_accuracy = model_vgg16.evaluate(val_subset)
print(f"Validation Loss For DAMAGED vs UNDAMAGED Dataset using VGG16 is: {val_loss}")
print(f"Validation Accuracy For DAMAGED vs UNDAMAGED Dataset using VGG16 is: {val_accuracy}")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Generate predictions on the validation dataset
y_pred_prob = model_vgg16.predict(val_ds)
y_pred = np.where(y_pred_prob > 0.5, 1, 0).flatten()  # Convert probabilities to binary labels

# Extract true labels from the validation dataset
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# Convert y_true to binary labels if needed
y_true_binary = np.where(y_true > 0, 1, 0)

# Make sure y_pred and y_true_binary have the same length
min_length = min(len(y_pred), len(y_true_binary))
y_pred = y_pred[:min_length]
y_true_binary = y_true_binary[:min_length]

# Calculate precision, recall, and F1 score
precision = precision_score(y_true_binary, y_pred)
recall = recall_score(y_true_binary, y_pred)
f1 = f1_score(y_true_binary, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Extract the training and validation loss and accuracy
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, 'b', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracy, 'b', label='Training accuracy')
plt.plot(epochs_range, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as a JPEG image
plt.savefig('training_validation_plots.jpg', format='jpg')
plt.show()
