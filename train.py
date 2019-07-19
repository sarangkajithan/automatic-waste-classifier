from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


num_classes = 3
resnet_weights_path = 'C:/Users/USER/Desktop/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 60
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        'C:/Users/USER/Desktop/resnet50/images/train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'C:/Users/USER/Desktop/resnet50/images/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

history=my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs = 10,
        validation_data = validation_generator,
        validation_steps=100)
from keras.models import load_model

my_new_model.save_weights("C:/Users/USER/Desktop/resnet50/my_model.h5")  # creates a HDF5 file 'my_model.h5' 
Y_pred = my_new_model.predict_generator(validation_generator, 364  // 25)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Cardboard', 'paper', 'plastic']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

