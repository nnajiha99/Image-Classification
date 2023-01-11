
# Image Classification

This project is intended to perform image classification to classify concretes with or without cracks. 

The dataset is obtained from https://data.mendeley.com/datasets/5y9wdsg2zt/2. The data comes as a .rar file, which contains concrete images having cracks. The dataset is divided into two: positive and crack images for image classification.
Each class has 20000 images with a total of 40000 images. For this project, I am using Google Colab to train the model.


## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)



## Details of Steps

In order to download the dataset inside the Google Colab environment without uploading to Google Colab, the following code is implemented:

    !wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip
    
To unzip the file:

    !unzip "5y9wdsg2zt-2.zip" 

To unrar the file:

    !unrar x 'Concrete Crack Images for Classification.rar'


Import all packages

    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    from tensorflow.keras import layers,optimizers,losses,metrics,callbacks,applications
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

- Data Loading

    Load the data. In dataset folder distributes into two folders: positive and negative.

        PATH = os.path.join(os.getcwd(), 'dataset')

    Use spilt-folders library to split the folders into 3 sets: train, validation and test. We need to specify the folder directory that we want to split and the output folder. 
    The ratio of split has been set as 70% for train, 20% for validation and 10% for test.

        !pip install split-folders
        splitfolders.ratio('/content/dataset', output="data", seed=1337, ratio=(.7, 0.2,0.1)) 

- Data Preparation

    Define the path to the train and validation data folder.

        train_path = '/content/data/train'
        val_path = '/content/data/val'
        test_path = '/content/data/test'

    Define the batch size and image size.
        
        BATCH_SIZE = 32
        IMG_SIZE = (160,160)

    Load the data into tensorflow dataset using the specific method.

        train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
        val_dataset = keras.utils.image_dataset_from_directory(val_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)

    To visualize some images.

        class_names = train_dataset.class_names

        plt.figure(figsize=(10,10))
        for images, labels in train_dataset.take(1):
            for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')

    Further split the validation dataset into validation-test split.

        val_batches = tf.data.experimental.cardinality(val_dataset)
        test_dataset = val_dataset.take(val_batches//5)
        validation_dataset = val_dataset.skip(val_batches//5)

    Convert the BatchDataset into PrefetchDataset.

        AUTOTUNE = tf.data.AUTOTUNE

        pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
        pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

    Create a small pipeline for data augmentation.

        data_augmentation = keras.Sequential()
        data_augmentation.add(layers.RandomFlip('horizontal'))
        data_augmentation.add(layers.RandomRotation(0.2))

    Display data augmentation.

        for images,labels in pf_train.take(1):
            first_image = images[0]
            plt.figure(figsize=(10,10))
            for i in range(9):
                plt.subplot(3,3,i+1)
                augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
                plt.imshow(augmented_image[0]/255.0)
                plt.axis('off')

- Model Development

    Prepare the layer for data preprocessing. We use MobileNetV2 model to perform transfer learning on the dataset.

        preprocess_input = applications.mobilenet_v2.preprocess_input

    Apply transfer learning.

        IMG_SHAPE = IMG_SIZE + (3,)
        feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

    Disable the training for the feature extractor.

        feature_extractor.trainable = False
        feature_extractor.summary()
        keras.utils.plot_model(feature_extractor,show_shapes=True)

    Create the classification layers. Functional API is used to link all of the modules together.

        global_avg = layers.GlobalAveragePooling2D()
        output_layer = layers.Dense(len(class_names),activation='softmax')

        inputs = keras.Input(shape=IMG_SHAPE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = feature_extractor(x)
        x = global_avg(x)
        x = layers.Dropout(0.3)(x)
        outputs = output_layer(x)

        model = keras.Model(inputs=inputs,outputs=outputs)
        model.summary()

    After creating the network, we need to compile the model before fitting it to the dataset.

        optimizer = optimizers.Adam(learning_rate=0.0001)
        loss = losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

    We try to evaluate the model before model training.

        loss0,accuracy0 = model.evaluate(pf_val)
        print("Loss = ",loss0)
        print("Accuracy = ",accuracy0)

    Train the model.
    
        log_path = os.path.join('log_dir','tl_demo',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tb = callbacks.TensorBoard(log_dir=log_path)
        es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

        history = model.fit(pf_train,validation_data=pf_val,epochs=5,callbacks=[tb,es])

- Model Deployment

    Visualize the model training performance.

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

    Evaluate the final model.

        test_loss,test_acc = model.evaluate(pf_test)
        print("Loss = ",test_loss)
        print("Accuracy = ",test_acc)

    Deploy the model using the test data. Then, compare label and prediction.

        image_batch, label_batch = pf_test.as_numpy_iterator().next()
        predictions = np.argmax(model.predict(image_batch),axis=1)

        label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))

- Model Saving

    To save trained model.

        model.save('model.h5')

    To save model architecture.

        plot_model(model, to_file='model.png')

## Model Performances

- Image Visualization

    ![images](https://user-images.githubusercontent.com/121777112/211855544-632ed4ef-4a3b-47ec-8a05-167a145e8164.png)

- Data Augmentation

    ![data_augmentation](https://user-images.githubusercontent.com/121777112/211855625-b2551e03-b395-4b22-88ec-eb5721122db0.png)

- Model Analysis

    ![training_accuracy_and_loss](https://user-images.githubusercontent.com/121777112/211855836-339e3c73-9a2a-4992-ac31-1c4c04b5b67d.png)

- Training Performance

    Accuracy before training:

    ![accuracy_before_train](https://user-images.githubusercontent.com/121777112/211855717-cd12f9ca-0a69-491d-8508-d88bb1734924.png)

    Accuracy after training:

    ![accuracy_after_train](https://user-images.githubusercontent.com/121777112/211855799-94f32655-035c-48d7-b29e-e797b57d2cc4.png)

- Model Architecture

    ![model_architecture](https://user-images.githubusercontent.com/121777112/211855665-06ccb1ae-f7a5-4648-b94c-b269cccb4263.png)
    
## Discussion

With the test accuracy of 99.75%, there are only a few classification errors occurred out of the 4000 test images. The model seems not overfitting.
## Acknowledgements

 - [dataset source](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

