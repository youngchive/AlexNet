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


# 디렉토리명과 파일명을 통해서 레이블
DATASET_DIR = 'Alphabet_Dataset'
#DATASET_DIR = 'Rice_Dataset'
DATASET_DIR_LEN = len(DATASET_DIR)
DATASET_TYPE = '.jpg'

IMAGE_SIZE = 227
CLASSES = 3
COLOR = 3  # 컬러는 3, 흑백은 1
BATCH_SIZE = 32  # 배치사이즈
REGULARIZER = 1e-5
LEARNING_RATE = 1e-5
EPOCH = 10

TEST_SIZE = 0.2
VALID_SIZE = 0.2


def make_alphatype_dataframe(dataset_dir=DATASET_DIR):
    paths = []
    label_gubuns = []
    # walk(): 하위의 폴더들을 for문으로 탐색할 수 있게 해준다.
    for dirname, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            # 이미지 파일이 아닌 파일도 해당 디렉토리에 있음.
            if DATASET_TYPE in filename:
                # 파일의 절대 경로를 file_path 변수에 할당.
                filename = dirname+'/'+ filename
                paths.append(filename)
                # 이미지 파일의 절대 경로에서 레이블명 생성을 위한 1차 추출. '/'로 분할하여 파일 바로 위 서브디렉토리 이름 가져옴.
                start_pos = filename.find('/', DATASET_DIR_LEN-1) # '/' 있는 인덱스
                end_pos = filename.rfind('/')
                alphatype = filename[start_pos+1:end_pos] # 디렉토리 이름
                label_gubuns.append(alphatype)

    data_df = pd.DataFrame({'path':paths, 'label':label_gubuns})
    return data_df


pd.set_option('display.max_colwidth',200) # column의 width 설정
data_df = make_alphatype_dataframe()
print('data_df shape:', data_df.shape)
data_df.head() # DataFrame 앞 5개 출력
data_df['label'].value_counts() # 개별 분포도 확인





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


# 입력 인자 images_array labels는 모두 numpy array로 들어옴.
# 인자로 입력되는 images_array는 전체 32x32 image array임.


class Alphabet_Dataset(Sequence):
    def __init__(self, images_array, labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=None):
        '''
        파라미터 설명
        images_array: 원본 64*64 만큼의 image 배열값.
        labels: 해당 image의 label들
        batch_size: __getitem__(self, index) 호출 시 마다 가져올 데이터 batch 건수
        augmentor: albumentations 객체
        shuffle: 학습 데이터의 경우 epoch 종료시마다 데이터를 섞을지 여부
        '''
        # 객체 생성 인자로 들어온 값을 객체 내부 변수로 할당.
        # 인자로 입력되는 images_array는 전체 32x32 image array임.
        self.images_array = images_array
        self.labels = labels
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.pre_func = pre_func
        # train data의 경우
        self.shuffle = shuffle
        if self.shuffle:
            # 객체 생성시에 한번 데이터를 섞음.
            # self.on_epoch_end()
            pass

    # Sequence를 상속받은 Dataset은 batch_size 단위로 입력된 데이터를 처리함.
    # __len__()은 전체 데이터 건수가 주어졌을 때 batch_size단위로 몇번 데이터를 반환하는지 나타남
    def __len__(self):
        # batch_size단위로 데이터를 몇번 가져와야하는지 계산하기 위해 전체 데이터 건수를 batch_size로 나누되, 정수로 정확히 나눠지지 않을 경우 1회를 더한다.
        return int(np.ceil(len(self.labels) / self.batch_size))

    # batch_size 단위로 image_array, label_array 데이터를 가져와서 변환한 뒤 다시 반환함
    # 인자로 몇번째 batch 인지를 나타내는 index를 입력하면 해당 순서에 해당하는 batch_size 만큼의 데이타를 가공하여 반환
    # batch_size 갯수만큼 변환된 image_array와 label_array 반환.
    def __getitem__(self, index):
        # index는 몇번째 batch인지를 나타냄.
        # batch_size만큼 순차적으로 데이터를 가져오려면 array에서 index*self.batch_size:(index+1)*self.batch_size 만큼의 연속 데이터를 가져오면 됨
        # 32x32 image array를 self.batch_size만큼 가져옴.
        images_fetch = self.images_array[index * self.batch_size:(index + 1) * self.batch_size]
        if self.labels is not None:
            label_batch = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # 만일 객체 생성 인자로 albumentation으로 만든 augmentor가 주어진다면 아래와 같이 augmentor를 이용하여 image 변환
        # albumentations은 개별 image만 변환할 수 있으므로 batch_size만큼 할당된 image_name_batch를 한 건씩 iteration하면서 변환 수행.
        # 변환된 image 배열값을 담을 image_batch 선언. image_batch 배열은 float32 로 설정.
        image_batch = np.zeros((images_fetch.shape[0], IMAGE_SIZE, IMAGE_SIZE, COLOR), dtype='float32')

        # batch_size에 담긴 건수만큼 iteration 하면서 opencv image load -> image augmentation 변환(augmentor가 not None일 경우)-> image_batch에 담음.
        for image_index in range(images_fetch.shape[0]):
            # image = cv2.cvtColor(cv2.imread(image_name_batch[image_index]), cv2.COLOR_BGR2RGB)
            # 원본 image를 IMAGE_SIZE x IMAGE_SIZE 크기로 변환
            image = cv2.resize(images_fetch[image_index], (IMAGE_SIZE, IMAGE_SIZE))
            # 만약 augmentor가 주어졌다면 이를 적용.
            if self.augmentor is not None:
                image = self.augmentor(image=image)['image']

            # 만약 scaling 함수가 입력되었다면 이를 적용하여 scaling 수행.
            if self.pre_func is not None:
                image = self.pre_func(image)

            # image_batch에 순차적으로 변환된 image를 담음.
            image_batch[image_index] = image

        return image_batch, label_batch

    # epoch가 한번 수행이 완료 될 때마다 모델의 fit()에서 호출됨.
    def on_epoch_end(self):
        if (self.shuffle):
            # print('epoch end')
            # 원본 image배열과 label를 쌍을 맞춰서 섞어준다. scikt learn의 utils.shuffle에서 해당 기능 제공
            self.images_array, self.labels = sklearn.utils.shuffle(self.images_array, self.labels)
        else:
            pass


def zero_one_scalar(image):
    return image/255.0

tr_ds = Alphabet_Dataset(tr_images, tr_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=True, pre_func=zero_one_scalar)
val_ds = Alphabet_Dataset(val_images, val_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=zero_one_scalar)

model = create_alexnet(in_shape=(IMAGE_SIZE, IMAGE_SIZE, COLOR), n_classes=CLASSES,
                       kernel_regular=regularizers.l2(l2=REGULARIZER), optimizer=Adam(learning_rate=LEARNING_RATE))

# 5번 iteration내에 validation loss가 향상되지 않으면 learning rate을 기존 learning rate * 0.2로 줄임.
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='min', verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

steps_per_epoch = tr_images.shape[0] // BATCH_SIZE
validation_steps = val_images.shape[0] // BATCH_SIZE
if tr_images.shape[0] % BATCH_SIZE != 0:
    steps_per_epoch += 1
if val_images.shape[0] % BATCH_SIZE != 0:
    validation_steps += 1

with tf.device("/device:GPU:0"):
    history = model.fit(tr_ds, epochs=EPOCH,
                        #steps_per_epoch=steps_per_epoch,
                        validation_data=val_ds,
                        #validation_steps=validation_steps,
                        callbacks=[rlr_cb, ely_cb]
                        )


test_ds = Alphabet_Dataset(test_images, test_oh_labels, batch_size=BATCH_SIZE, augmentor=None, shuffle=False, pre_func=zero_one_scaler)
model.evaluate(test_ds)
