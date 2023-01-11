
!wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip

!unzip "5y9wdsg2zt-2.zip"

!unrar x 'Concrete Crack Images for Classification.rar'

from tensorflow.keras import layers,optimizers,losses,metrics,callbacks,applications
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import os, datetime
import splitfolders
import pandas as pd
import numpy as np

#1. Data Loading
PATH = os.path.join(os.getcwd(), 'dataset')

!pip install split-folders

splitfolders.ratio('/content/dataset', output="data", seed=1337, ratio=(.7, 0.2,0.1))

#2. Data preparation
#(A) Define the path to the train and validation data folder
train_path = '/content/data/train'
val_path = '/content/data/val'
test_path = '/content/data/test'

#(B) Define the batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)

#(C) Load the data into tensorflow dataset using the specific method
train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(val_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

#3. Display some images as example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
    for i in range(9):
      plt.subplot(3,3,i+1)
      plt.imshow(images[i].numpy().astype('uint8'))
      plt.title(class_names[labels[i]])
      plt.axis('off')

#4. Further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#5. Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#6. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#Apply the data augmentation to test it out
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
      plt.subplot(3,3,i+1)
      augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
      plt.imshow(augmented_image[0]/255.0)
      plt.axis('off')

#7. Prepare the layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

#8. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

#9. Create the classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names),activation='softmax')

#10. Use functional API to link all of the modules together
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#11. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#Evaluate the model before model training
loss0,accuracy0 = model.evaluate(pf_val)
print("Loss = ",loss0)
print("Accuracy = ",accuracy0)

log_path = os.path.join('log_dir','tl_demo',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)
es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

#12. Train the model
history = model.fit(pf_train,validation_data=pf_val,epochs=5,callbacks=[tb,es])

#Visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(acc, label='train accuracy')
plt.plot(val_acc, label='val accuracy')
plt.title('epoch_accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.title('epoch_loss')
plt.legend()
plt.show()

#13. Evaluate the final model
test_loss,test_acc = model.evaluate(pf_test)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)

#14. Model deployment
#Deploy the model using the test data
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

#Compare label and prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

#15. Model saving

#to save trained model
model.save('model.h5')

#To save model architecture
plot_model(model, to_file='model.png')