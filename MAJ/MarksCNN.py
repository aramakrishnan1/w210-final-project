import os, shutil, argparse, random, pickle, cv2, fnmatch, PIL, math, signal, warnings
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings("ignore")


print('[INFO] Imports done.  Using Tensorflow version: {}, and Keras version: {}'.format(tf.__version__, k.__version__))

config={}

config['do_dataprep'] = False

config['source'] = os.path.join(os.getcwd(), 'source_data')
config['home_dir'] = os.path.join(os.getcwd(), 'data_binary_split')

config['train_dir'] = os.path.join(os.getcwd(), 'data_binary_split', 'train')
config['test_dir'] = os.path.join(os.getcwd(), 'data_binary_split', 'test')
config['val_dir'] = os.path.join(os.getcwd(), 'data_binary_split', 'validation')

config['train_pain'] = os.path.join(config['train_dir'], 'pain')
config['train_nopain'] = os.path.join(config['train_dir'], 'nopain')
config['test_pain'] = os.path.join(config['test_dir'], 'pain')
config['test_nopain'] = os.path.join(config['test_dir'], 'nopain')
config['val_pain'] = os.path.join(config['val_dir'], 'pain')
config['val_nopain'] = os.path.join(config['val_dir'], 'nopain')

config['val_split'] = 0.1
config['test_split'] = 0.1

config['target_size'] = (100,100)# (320, 240)

config['train_batch'] = 100
config['test_batch'] = 100
config['val_batch'] = 100

config['epochs'] = 100

def split(SOURCE, TRAINING, TESTING, VALIDATION):
    
    print('[INFO] In Splitting')

    files = []
    
    for file in os.listdir(SOURCE): 
        files.append(file)

    training_length = int(len(files) * (1-config['val_split']-config['test_split']))
    testing_length = int(len(files) * config['test_split'])
    validation_length = int(len(files) * config['val_split'])

    shuffled_set = random.sample(files, len(files))

    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:training_length+testing_length]
    validation_set = shuffled_set[training_length+testing_length:]

    for filename in training_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING, filename)
        shutil.copyfile(this_file, destination)

    for filename in testing_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TESTING, filename)
        shutil.copyfile(this_file, destination)

    for filename in validation_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(VALIDATION, filename)
        shutil.copyfile(this_file, destination)

new_dirs = [
    config['home_dir']
    , config['train_dir'], config['test_dir'], config['val_dir']
    , config['train_pain'], config['train_nopain']
    , config['test_pain'], config['test_nopain']
    , config['val_pain'], config['val_nopain']
           ]

if config['do_dataprep']: 
    if os.path.exists(config['home_dir']):
        shutil.rmtree(config['home_dir'])
        
    for value in new_dirs: 
        if not os.path.exists(value):
            os.mkdir(value)
    pain_data_location = os.path.join(config['source'], '0.0')

    print('[INFO] Files in pain origin dir: \t{}'.format(len(os.listdir(pain_data_location))))
    print('[INFO] Files in pain train dir: \t{}'.format(len(os.listdir(config['train_pain']))))
    print('[INFO] Files in pain test dir: \t\t{}'.format(len(os.listdir(config['test_pain']))))
    print('[INFO] Files in pain val dir: \t\t{}'.format(len(os.listdir(config['val_pain']))))
    
    split(
        SOURCE = pain_data_location
        , TRAINING = config['train_pain'] 
        , TESTING = config['test_pain'] 
        , VALIDATION = config['val_pain'] 
    )

    print('[INFO] Files in pain origin dir: \t{}'.format(len(os.listdir(pain_data_location))))
    print('[INFO] Files in pain train dir: \t{}'.format(len(os.listdir(config['train_pain']))))
    print('[INFO] Files in pain test dir: \t\t{}'.format(len(os.listdir(config['test_pain']))))
    print('[INFO] Files in pain val dir: \t\t{}'.format(len(os.listdir(config['val_pain']))))
    
    print('[INFO] Files in nopain train dir: \t{}'.format(len(os.listdir(config['train_nopain']))))
    print('[INFO] Files in nopain test dir: \t{}'.format(len(os.listdir(config['test_nopain']))))
    print('[INFO] Files in nopain val dir: \t{}'.format(len(os.listdir(config['val_nopain']))))
    
    nopain_folders = os.listdir(config['source'])
    nopain_folders.remove('0.0')
    
    for folder in nopain_folders:
        split(
            SOURCE = os.path.join(config['source'], folder)
            , TRAINING = config['train_nopain'] 
            , TESTING = config['test_nopain'] 
            , VALIDATION = config['val_nopain'] 
        )

    print('[INFO] Files in pain train dir: \t{}'.format(len(os.listdir(config['train_pain']))))
    print('[INFO] Files in pain test dir: \t\t{}'.format(len(os.listdir(config['test_pain']))))
    print('[INFO] Files in pain val dir: \t\t{}'.format(len(os.listdir(config['val_pain']))))

def create_and_compile_model(): 
    model = k.models.Sequential([
        k.layers.Conv2D(16, (3,3), activation='relu', input_shape=(*config['target_size'], 3),  name='mh-conv-1')
        , k.layers.MaxPooling2D(2, 2, name='mh-maxpool-1')
        , k.layers.Conv2D(32, (3,3), activation='relu', name='mh-conv-2')
        , k.layers.MaxPooling2D(2,2, name='mh-maxpool-2')
        , k.layers.Conv2D(64, (3,3), activation='relu', name='mh-conv-3')
        , k.layers.MaxPooling2D(2,2, name='mh-maxpool-3')
        , k.layers.Conv2D(128, (3,3), activation='relu', name='mh-conv-4')
        , k.layers.MaxPooling2D(4,4, name='mh-maxpool-4')
        , k.layers.Flatten(name='mh-flatten-1')
        , k.layers.Dense(128, activation='relu', name='mh-dense-1')
        , k.layers.Dense(1, activation='sigmoid', name='mh-dense-output')
    ])
    
    model.compile(
        optimizer=Adam()
        , loss='binary_crossentropy'
        , metrics=['accuracy']
    )
    
    return model

new_directory = os.path.join(os.getcwd(), 'trial01')
if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
os.mkdir(new_directory)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print('\nReached 99% accuracy which is satisfactory so stopped training!')
            self.model.stop_training = True
        if(logs.get('acc')<0.4):
            print('\nUnable to go over 40% accuracy, so cancelling run!')
            self.model.stop_training = True

markscallbacks = myCallback()

config['train_count'] = sum([len(files) for r, d, files in os.walk(config['train_dir'])])
config['train_steps'] = math.ceil(config['train_count']/config['train_batch'])

train_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    config['train_dir']
    , target_size = config['target_size'] 
    , batch_size = config['train_batch']
    , class_mode = 'binary'
)

config['val_count'] = sum([len(files) for r, d, files in os.walk(config['val_dir'])])
config['val_steps'] = math.ceil(config['val_count']/config['val_batch'])

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

val_generator = val_datagen.flow_from_directory(
    config['val_dir']  # This is the source directory for training images
    , target_size = config['target_size']  # All images will be resized to 150x150 for compressing
    , batch_size = config['val_batch']
    , class_mode = 'binary' # Since we use binary_crossentropy loss, we need binary labels
)

config['test_count'] = sum([len(files) for r, d, files in os.walk(config['test_dir'])])
config['test_steps'] = math.ceil(config['test_count']/config['test_batch'])

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    config['test_dir']  # This is the source directory for training images
    , target_size = config['target_size']  # All images will be resized to 150x150 for compressing
    , batch_size = config['test_batch']
    , class_mode = 'binary' # Since we use binary_crossentropy loss, we need binary labels
)

model = create_and_compile_model()

model.summary()

history = model.fit_generator(
    train_generator
    , steps_per_epoch = config['train_steps']
    , epochs = config['epochs']
    , validation_data = val_generator
    , validation_steps = config['val_steps']
    , callbacks=[
        markscallbacks
        , EarlyStopping(monitor='acc', patience=5)
        , ModelCheckpoint(filepath=os.path.join(new_directory, 'model.h5'), monitor='acc', save_best_only=True)
        , ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]
)

print('Accuracy on test data: {0:2.2f}%'.format(100*model.evaluate_generator(test_generator, steps=config['test_steps'])[1]))

res_dict = {
    'epochs' : config['epochs']
    , 'accuracy' : history.history['acc']
    , 'val_accuracy' : history.history['val_acc']
    , 'loss' : history.history['loss']
    , 'val_loss' : history.history['val_loss']
}

df = pd.DataFrame(res_dict)
df.to_csv(os.path.join(new_directory, '01 - results.csv'))

os.kill(os.getpid(), signal.SIGKILL)