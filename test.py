
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
num_classes = 3
resnet_weights_path = 'C:/Users/USER/Desktop/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.models import load_model
my_new_model.load_weights('C:/Users/USER/Desktop/resnet50/my_model.h5')

import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 50
img_paths= 'C:/Users/USER/Desktop/resnet50/test.jpg'
img_height=image_size,
img_width=image_size
imgs = load_img(img_paths)
img_array = np.array([img_to_array(imgs)])
output = preprocess_input(img_array)
result = my_new_model.predict_classes(output)
result
if(result == 2):
    print("plastic bottle!")
elif(result == 1):
    print("paper cup!")
elif(result == 0):
   print("cardboard!")
