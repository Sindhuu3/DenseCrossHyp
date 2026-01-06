import numpy as np
import tensorflow as tf

def preprocess_image(image, img_size):
    image = image.resize(img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0
    return img_arr
