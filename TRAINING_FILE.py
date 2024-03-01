# %%
#1. Load Necessary Libraries
import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import optimizers
from keras import layers, optimizers, losses, callbacks, models, applications
from keras.api._v2.keras.applications import MobileNetV2

# %%
#2. Loading Data
path = os.path.join(os.getcwd(), 'Dataset')
IMG_SIZE = (256,256)
BATCH_SIZE = 128
SEED = 12345
train_dataset = tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2, subset="training", seed=SEED, image_size = IMG_SIZE, batch_size = BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2, subset="validation", seed=SEED, image_size = IMG_SIZE, batch_size = BATCH_SIZE)
class_names = ['not-cracked', 'cracked']

# %%
# Splitting data to test data
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
# Data Visualization
class_names = val_dataset.class_names
plt.figure(figsize=(10,10))
for images, labels in val_dataset.take(1):
  for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# %%
# Converting tensorflow datasets into prefetch Dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip('horizontal'),
#   tf.keras.layers.RandomRotation(0.2),
#   tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
# ])

# %%
# layer to perform data normalization
import keras.applications.mobilenet_v2 as mobilenet_v2
preprocess_input = mobilenet_v2.preprocess_input

# %%
# Perform Transfer Learning
#(A) Load the pretrained model as feature 
IMG_SHAPE = IMG_SIZE + (3,)
base_model = MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights='imagenet')
base_model.summary()
base_model.trainable = False

# %%
# Define the classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(1,activation ='sigmoid')

# %%
#10. Build the entire model pipeline
#(A) Input
inputs = keras.Input(shape = IMG_SHAPE)
#(B) Data Normalization layer
x = preprocess_input(inputs)
#(C) Transfer Learning feature Extractor
x = base_model(x,training=False)
#(D) Classification Layers
x = global_avg(x)
x = layers.Dropout(0.2)(x)
outputs = output_layer(x)
#(E) Define the full model out
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

# %%
# Compiling Model
optimizer = optimizers.Adam(learning_rate = 0.00001)
loss = losses.BinaryCrossentropy()
model.compile(optimizer = optimizer, loss = loss, metrics =['accuracy'])


# %%
# Prepare the callback functions for model training
early_stopping = callbacks.EarlyStopping(patience = 3)


# %%
# Model classification layer training
logpath = os.path.join('tensorboard_log', datetime.datetime.now().strftime('%Y%m%d - %H%M%S'))
tb = callbacks.TensorBoard(logpath)
EPOCH = 15
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = EPOCH,
    callbacks = [early_stopping, tb]
)

# %%
len(base_model.layers)
#num_layers = len(base_model.layers)

# %%
#14. Model Fine tuning by training the top layers of the base model along with the classifier
base_model.trainable = True

for layer in base_model.layers[:144]:
    layer.trainable = False

base_model.summary()

# %%
optimizer = keras.optimizers.RMSprop(learning_rate = 0.00001)
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
model.summary()

# %%
fine_tune_epochs = 10
total_epochs =  EPOCH + fine_tune_epochs
loss = losses.BinaryCrossentropy()
history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks = [tb,early_stopping]
                         )

# %%
# Save the model after training
model.save("my_trained_model.h5")

# %%
#Plot the trainig graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# %%
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([EPOCH-1,EPOCH-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([EPOCH-1,EPOCH-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %%
#Model Evaluation
test_loss,test_accuracy = model.evaluate(test_dataset)

print('Test Result')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')


