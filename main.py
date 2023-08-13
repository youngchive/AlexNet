import os
import numpy as np
import pandas as pd
import cv2
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, Lambda, ZeroPadding2D, MaxPooling2D , GlobalAveragePooling2D
from keras.optimizers import Adam , RMSprop
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler
from keras import regularizers
from PIL import Image
from keras.utils import Sequence
from keras.utils import to_categorical

# input shape, classes 개수, kernel_regularizer등을 인자로 가져감.
def create_alexnet(in_shape=(IMAGE_SIZE, IMAGE_SIZE, COLOR), n_classes=CLASSES, kernel_regular=regularizers.l2(l2=0.0005), optimizer='adam'):
    input_tensor = Input(shape=in_shape)

    # Layer 1
    x = Lambda(lambda image: tf.image.per_image_standardization(image))(input_tensor)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(11, 11), strides=(1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Layer 2
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=kernel_regular)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=kernel_regular)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Layer 4
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Layer 5
    x = Dense(units=10, activation='relu')(x)
    x = Dense(units=n_classes, activation='softmax')(x)

    # Model compilation and summary
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


# Alphabet 데이터 재 로딩 및 Scaling/OHE 전처리 적용하여 학습/검증/데이터 세트 생성.
train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, stratify=data_df['label'], random_state=2021)
print(train_df.shape, test_df.shape)

# DataFrame에서 numpy array로 변환.
train_path = train_df['path'].values
train_path_labels = pd.get_dummies(train_df['label']).values
test_path = test_df['path'].values
test_path_labels = pd.get_dummies(test_df['label']).values
print(train_path.shape, train_path_labels.shape)
print(train_path)

number_of_train_data = train_path.size  # 데이터 수
number_of_test_data = test_path.size


# 곱셈의 형태로 ndarray 만들기
train_images = np.zeros(number_of_train_data*IMAGE_SIZE*IMAGE_SIZE*COLOR, dtype=np.int32).reshape(number_of_train_data, IMAGE_SIZE, IMAGE_SIZE, COLOR)
train_labels = np.zeros(number_of_train_data, dtype=np.int32)
test_images = np.zeros(number_of_test_data*IMAGE_SIZE*IMAGE_SIZE*COLOR, dtype=np.int32).reshape(number_of_test_data, IMAGE_SIZE, IMAGE_SIZE, COLOR)
test_labels = np.zeros(number_of_test_data, dtype=np.int32)

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
# print(train_path_labels)


# PIL을 활용한 방법
i = 0
j = 0
file_path = train_path
for file in file_path:
    img = np.empty(shape=[IMAGE_SIZE, IMAGE_SIZE, COLOR])

    if(COLOR == 1): # 'L': greyscale
        img = np.array(Image.open(file).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.int32)
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, COLOR)
    elif(COLOR == 3):
        img = np.array(Image.open(file).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.int32) # 'RGB': 색상

    if i%1000 == 0:
        print("Train Completing... ", i)
    train_images[i, :, :, :] = img  # i번째에 이미지 픽셀값 입력

    while(1):
        if(train_path_labels[i][j]==1):
            break
        else:
            j+=1
    train_labels[i] = j
    i += 1
    j = 0

train_images = train_images/255.0  # 이미지 정규화

# PIL을 활용한 방법
i = 0
j = 0
file_path = test_path
for file in file_path:
    img = np.empty(shape=[IMAGE_SIZE, IMAGE_SIZE, COLOR])

    if (COLOR == 1):  # 'L': greyscale
        img = np.array(Image.open(file).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.int32)
        img = img.reshape(IMAGE_SIZE, IMAGE_SIZE, COLOR)
    elif (COLOR == 3):
        img = np.array(Image.open(file).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.int32)  # 'RGB': 색상

    if i % 100 == 0:
        print("Test Completing... ", i)

    test_images[i, :, :, :] = img  # i번째에 이미지 픽셀값 입력
    while(1):
        if(test_path_labels[i][j]==1):
            break
        else:
            j+=1
    test_labels[i] = j
    i += 1
    j = 0

test_images = test_images/255.0  # 이미지 정규화


def zero_one_scaler(image):
    return image / 255.0


def get_preprocessed_ohe(images, labels, pre_func=None):
    # preprocessing 함수가 입력되면 이를 이용하여 image array를 scaling 적용.
    if pre_func is not None:
        images = pre_func(images)
    # OHE 적용
    oh_labels = to_categorical(labels)
    return images, oh_labels


# 학습/검증/테스트 데이터 세트에 전처리 및 OHE 적용한 뒤 반환
def get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=VALID_SIZE,
                             random_state=2021):
    train_images_pre, train_oh_labels_pre = get_preprocessed_ohe(train_images, train_labels)
    test_images, test_oh_labels = get_preprocessed_ohe(test_images, test_labels)

    tr_images, val_images, tr_oh_labels, val_oh_labels = train_test_split(train_images_pre, train_oh_labels_pre,
                                                                          test_size=valid_size,
                                                                          random_state=random_state)

    # Split training Alphabet_Dataset into two subsets: main training set and additional validation set
    train_size = tr_images.shape[0]
    val_size = int(train_size * valid_size)
    tr_size = train_size - val_size

    tr_idx = np.random.choice(train_size, tr_size, replace=False)
    val_idx = np.setdiff1d(np.arange(train_size), tr_idx)[:val_size]

    tr_images, tr_oh_labels = train_images_pre[tr_idx], train_oh_labels_pre[tr_idx]
    val_images, val_oh_labels = train_images_pre[val_idx], train_oh_labels_pre[val_idx]

    return (tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels)


(tr_images, tr_oh_labels), (val_images, val_oh_labels), (test_images, test_oh_labels ) = \
    get_train_valid_test_set(train_images, train_labels, test_images, test_labels, valid_size=VALID_SIZE, random_state=2021)


print(train_labels.shape)
print(tr_images.shape, tr_oh_labels.shape, val_images.shape, val_oh_labels.shape, test_images.shape, test_oh_labels.shape)
