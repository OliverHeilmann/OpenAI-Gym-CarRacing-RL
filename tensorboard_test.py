import tensorflow as tf
import datetime
import os

############################## SERVER CONFIGURATION ##################################

# Prevent tensorflow from allocating the all of GPU memory
# From: https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)   # set memory growth option
    # tf.config.set_logical_device_configuration( gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=3000)] )    # set memory limit to 3 GB

# Creates a virtual display for OpenAI gym
# pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# Where are models saved? How frequently e.g. every x1 episode?
USERNAME                = "oah33"
MODEL_TYPE              = "NN"
TIMESTAMP               = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
SAVE_TRAINING_FREQUENCY = 1
model_dir = f"./model/{USERNAME}/{MODEL_TYPE}/{TIMESTAMP}/"

# Setup TensorBoard model
log_dir = f"logs/fit/{TIMESTAMP}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


############################## MAIN CODE BODY ##################################

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Get Hardware list
hardware = tf.config.list_physical_devices(device_type=None)

# Assign GPU/ CPU
with tf.device('/GPU:0'):
    model.fit(x=x_train, 
            y=y_train, 
            epochs=50,
            validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback])

# Save model to appropriate dir, defined at start of code.
if not os.path.exists(model_dir):
        os.makedirs(model_dir)
model.save_weights(model_dir + "TEST_MODEL")