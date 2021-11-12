# https://www.kaggle.com/rifatahommed/ultrsound-nerve-segmentation-using-segnet/notebook?select=1_unet_ner_seg.hdf5

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
import cv2

from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import Input
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, UpSampling2D, Lambda, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, add, Reshape
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

K.set_image_data_format('channels_last')

# 추가
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

path = "../US_nerve/train/"
file_list = os.listdir(path)

train_image_tmp = []
train_mask_tmp = glob(path + '*_mask*')                 # ../US_nerve/train\\10_100_mask.tif ...

for i in train_mask_tmp: 
    train_image_tmp.append(i.replace('_mask', ''))      # ../US_nerve/train\\10_100.tif ...
    
train_image = []
train_mask = []

for j in train_image_tmp:
    temp = j.replace('\\','/')
    train_image.append(temp)

for h in train_mask_tmp:
    temp = h.replace('\\','/')
    train_mask.append(temp)
        
# print(train_image[:10],"\n" ,train_mask[:10])


'''
# Display the first image and mask of the first subject.
image1 = np.array(Image.open(path+"1_1.tif"))
image1_mask = np.array(Image.open(path+"1_1_mask.tif"))
image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)
fig, ax = plt.subplots(1,3,figsize = (16,12))
ax[0].imshow(image1, cmap = 'gray')
ax[1].imshow(image1_mask, cmap = 'gray')
ax[2].imshow(image1, cmap = 'gray', interpolation = 'none')
ax[2].imshow(image1_mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)
'''


width = 128
height = 128


# 평가 지표
def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)           
    y_pred_f = K.flatten(y_pred)            
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    # union = K.sum(y_true_f + y_pred_f - intersection) 
    return intersection/union

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky_index(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss( y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky_index(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def mish(x):
    return x * K.tanh(K.softplus(x))

get_custom_objects().update({'mish': mish})    


# 모델 구성
def segnet(input_size=(512, 512, 1)):
    kernel = 3  ####### 수정

    # Encoding layer
    img_input = Input(input_size)

    # Conv2D(필터의 수, (커널의 행, 열))
    x = Conv2D(64, (kernel, kernel), padding='same', name='conv1',strides= (1,1))(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (kernel, kernel), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(128, (kernel, kernel), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (kernel, kernel), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (kernel, kernel), padding='same', name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (kernel, kernel), padding='same', name='conv6')(x)
    x = BatchNormalization(name='bn6')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (kernel, kernel), padding='same', name='conv7')(x)
    x = BatchNormalization(name='bn7')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, (kernel, kernel), padding='same', name='conv8')(x)
    x = BatchNormalization(name='bn8')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='conv9')(x)
    x = BatchNormalization(name='bn9')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='conv10')(x)
    x = BatchNormalization(name='bn10')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(512, (kernel, kernel), padding='same', name='conv11')(x)
    x = BatchNormalization(name='bn11')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='conv12')(x)
    x = BatchNormalization(name='bn12')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (kernel, kernel), padding='same', name='conv13')(x)
    x = BatchNormalization(name='bn13')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    print("Build enceder done..")

    x = Dense(1024, activation = 'relu', name='fc1')(x)
    x = Dense(1024, activation = 'relu', name='fc2')(x)             

    # Decoding Layer 
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (kernel, kernel), padding='same', name='deconv1')(x)
    x = BatchNormalization(name='bn14')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (kernel, kernel), padding='same', name='deconv2')(x)
    x = BatchNormalization(name='bn15')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (kernel, kernel), padding='same', name='deconv3')(x)
    x = BatchNormalization(name='bn16')(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(512, (kernel, kernel), padding='same', name='deconv4')(x)
    x = BatchNormalization(name='bn17')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(512, (kernel, kernel), padding='same', name='deconv5')(x)
    x = BatchNormalization(name='bn18')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (kernel, kernel), padding='same', name='deconv6')(x)
    x = BatchNormalization(name='bn19')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(256, (kernel, kernel), padding='same', name='deconv7')(x)
    x = BatchNormalization(name='bn20')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(256, (kernel, kernel), padding='same', name='deconv8')(x)
    x = BatchNormalization(name='bn21')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (kernel, kernel), padding='same', name='deconv9')(x)
    x = BatchNormalization(name='bn22')(x)
    x = Activation('relu')(x)

    x = UpSampling2D()(x)
    x = Conv2DTranspose(128, (kernel, kernel), padding='same', name='deconv10')(x)
    x = BatchNormalization(name='bn23')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (kernel, kernel), padding='same', name='deconv11')(x)
    x = BatchNormalization(name='bn24')(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(64, (kernel, kernel), padding='same', name='deconv12')(x)
    x = BatchNormalization(name='bn25')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(1, (kernel, kernel), padding='same', name='deconv13')(x)
    x = BatchNormalization(name='bn26')(x)
    x = Activation('sigmoid')(x)
    pred = Reshape((height, width, 1))(x)
    print("Build decoder done..")
    
    return Model(inputs=img_input, outputs=pred)    


def train_generator(data_frame, batch_size, train_path, aug_dict,
        image_color_mode = "grayscale",
        mask_color_mode = "grayscale",
        image_save_prefix = "image",
        mask_save_prefix = "mask",
        save_to_dir = None,    
        target_size = (256,256),
        seed = 1):
        # save_to_dir = None
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    image_datagen과 mask_datagen에 대해 동일한 시드를 사용하여 생성자의 결과를 시각화하고 
    save_to_dir = "your path"를 설정하려면 image_dataagen과 mask_datagen에 대해 동일한 시드를 사용할 수 있습니다.
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen  = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size  = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size  = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img, mask)
  

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)   


pos_mask = []
pos_img  = []
neg_mask = []
neg_img  = []


for mask_path, img_path in zip(train_mask, train_image):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if np.sum(mask) == 0:                       # detection하는 게 없는 샘플들 -> neg
        neg_mask.append(mask_path)
        neg_img.append(img_path)
    else:
        pos_mask.append(mask_path)
        pos_img.append(img_path)


os.makedirs('../US_nerve/generated', exist_ok=True)
os.makedirs('../US_nerve/generated/img', exist_ok=True) 


def flip_up_down(img):
    newImg = img.copy()
    return cv2.flip(newImg, 0)      # 상하대칭

def flip_right_left(img):
    newImg = img.copy()
    return cv2.flip(newImg, 1)      # 좌우대칭


gen_img  = []
gen_mask = []


# data generation
for (img_path, mask_path) in tqdm(zip(pos_img, pos_mask)):
    image_name = img_path.split('/')[-1].split('.')[0]

    uf_img_path  = '../US_nerve/generated/img/'+image_name+'_uf.png'
    uf_mask_path = '../US_nerve/generated/img/'+image_name+'_uf_mask.png'
    rf_img_path  = '../US_nerve/generated/img/'+image_name+'_rf.png'
    rf_mask_path = '../US_nerve/generated/img/'+image_name+'_rf_mask.png'

    img  = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    uf_img  = flip_up_down(img)
    uf_mask = flip_up_down(mask)
    rf_img  = flip_right_left(img)
    rf_mask = flip_right_left(mask)

    cv2.imwrite(uf_img_path, uf_img)
    cv2.imwrite(uf_mask_path, uf_mask)
    cv2.imwrite(rf_img_path, rf_img)
    cv2.imwrite(rf_mask_path, rf_mask)
    
    # 2334it [00:06, 379.60it/s]

    # 생성 이미지 데이터 -> gen_img
    # 생성 마스크 데이터 -> gen_mask
    gen_img.append(uf_img_path)
    gen_mask.append(uf_mask_path)
    gen_img.append(rf_img_path)
    gen_mask.append(rf_mask_path)
    # 2323it [01:00, 38.59it/s]

print("Image Generation done")

aug_img  = gen_img + train_image                                    # aug_img = train_image + uf_img_path + rf_img_path
aug_mask = gen_mask + train_mask                                    # aug_mask = train_mask + uf_mask_path + rf_mask_path

df_ = pd.DataFrame(data={"filename": aug_img, 'mask' : aug_mask})   # 데이터프레임 생성. 데이터로 가져오는 것 : {"filename": aug_img, 'mask' : aug_mask}
df  = df_.sample(frac=1).reset_index(drop=True)                     # 데이터 샘플링. 데이터 전부(frac=1)를 가져오는데 reset_index로 기존 index가 아닌 새로운 indexing. 


kf = KFold(n_splits = 5, shuffle=False)                             # 5개의 fold로 분할


histories   = []
losses      = []
accuracies  = []
dicecoefs   = []
jacards     = []

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

EPOCHS = 300      
BATCH_SIZE = 32  ####### 수정


for k, (train_index, test_index) in enumerate(kf.split(df)):
    train_df = df.iloc[train_index]                 # train_index값을 가지는 데이터 추출 
    test_df = df.iloc[test_index]                   # test_index 값을 가지는 데이터 추출 

    test_data_frame = test_df.replace("\\","/")
    train_data_frame = train_df.replace("\\","/")


    train_gen = train_generator(train_data_frame, BATCH_SIZE,
                                None,
                                train_generator_args,
                                target_size=(height, width))

    test_gener = train_generator(test_data_frame, BATCH_SIZE,
                                None,
                                dict(),
                                target_size=(height, width))

    model = segnet(input_size=(height,width, 1))

    print(model.summary())
    
    model_checkpoint = ModelCheckpoint('bestmodel.hdf5', verbose=1, save_best_only=True)

    earlystopping = EarlyStopping(monitor='val_loss', patience=20)

    # 위치 변경
    # model = load_model(str(k+1)+'_ner_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'jacard': jacard, 'dice_coef': dice_coef})
    model = load_model('hyper.hdf5', custom_objects={'tversky_loss': tversky_loss, 'jacard': jacard, 'dice_coef': dice_coef})

    ####### 수정
    #model.compile(optimizer=Adam(lr=0.002), loss=dice_coef_loss, metrics=[jacard, dice_coef, 'binary_accuracy'])
    model.compile(optimizer=Nadam(lr=0.002), loss=tversky_loss, metrics=[jacard, dice_coef, 'binary_accuracy'])
    #model.compile(optimizer=Nadam(lr=0.002), loss=dice_coef_loss, metrics=[jacard, dice_coef, 'binary_accuracy'])
    #model.compile(optimizer=Adam(lr=1e-2), loss=tversky_loss, metrics=[jacard, dice_coef, 'binary_accuracy'])

    csv_logger = CSVLogger('./bestmodel.csv', append=True, separator=';')

    print("Model Training")
    # model.fit_generator -> model.fit
    history = model.fit(train_gen,
                        steps_per_epoch=len(train_data_frame) // BATCH_SIZE, 
                        epochs=EPOCHS, 
                        callbacks = [model_checkpoint, csv_logger, earlystopping],            # callbacks=[model_checkpoint, earlystopping],
                        validation_data = test_gener,
                        validation_steps = len(test_data_frame) // BATCH_SIZE) 
    
    model.save("bestmodel.hdf5")
    # print("save weight done..")

    test_gen = train_generator(test_data_frame, BATCH_SIZE,
                                None,
                                dict(),
                                target_size=(height, width))
    results = model.evaluate(test_gen, steps=len(test_data_frame) // BATCH_SIZE)
    results = dict(zip(model.metrics_names,results))
    
    histories.append(history)
    accuracies.append(results['binary_accuracy'])       # 훈련 정확도: 1에 가까울수록 정확도 높음
    losses.append(results['loss'])                      # 훈련 손실값: 결과값과의 차이이므로 0에 수렴할수록 좋음
    dicecoefs.append(results['dice_coef'])              
    jacards.append(results['jacard'])

    break


for h, history in enumerate(histories):
    
    keys = history.history.keys()
    fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))
    fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)

    for k, key in enumerate(list(keys)[:len(keys)//2]):
        training = history.history[key]
        validation = history.history['val_' + key]      

        epoch_count = range(1, len(training) + 1)

        axs[k].plot(epoch_count, training, 'r--')
        axs[k].plot(epoch_count, validation, 'b-')
        axs[k].legend(['Training ' + key, 'Validation ' + key])


print('average accuracy : ', np.mean(np.array(accuracies)), '+-', np.std(np.array(accuracies)))
print('average loss : ', np.mean(np.array(losses)), '+-', np.std(np.array(losses)))
print('average jacard : ', np.mean(np.array(jacards)), '+-', np.std(np.array(jacards)))
print('average dice_coef : ', np.mean(np.array(dicecoefs)), '+-', np.std(np.array(dicecoefs)))

#model = load_model('hyper.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'jacard': jacard, 'dice_coef': dice_coef})
#model = load_model('hyper.hdf5', custom_objects={'mish':mish, 'dice_coef_loss': dice_coef_loss, 'jacard': jacard, 'dice_coef': dice_coef})
model = load_model('bestmodel.hdf5', custom_objects={'tversky_loss': tversky_loss, 'jacard': jacard, 'dice_coef': dice_coef})
# model = load_model('0902_ner_tversky_seg.hdf5', custom_objects={'tversky_loss': tversky_loss, 'jacard': jacard, 'dice_coef': dice_coef})

np.random.seed(1)
index=np.random.randint(len(test_data_frame.index))

for i in range(25):   
    # index=np.random.randint(0,len(test_data_frame.index))    
    img = cv2.imread(test_data_frame['filename'].iloc[index], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (height, width))
    img = img[np.newaxis, :, :, np.newaxis]
    img = img / 255
    pred = model.predict(img)

    print(i+1, index)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.resize(cv2.imread(test_data_frame['mask'].iloc[index]), (height, width))))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    index = index + 1
    plt.show()