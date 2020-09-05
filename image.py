##################################
# 이미지 분류 튜토리얼
##################################

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


# 사진 데이터 캐싱 함수
# arguments: ()
# return: data_dir (텐서플로우 파일 객체?)
def load_data():
  # 주어진 url에서 사진 데이터 가져오기
  dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
  data_dir = pathlib.Path(data_dir)

  # 데이터의 각 꽃 개수 보기
  image_count = len(list(data_dir.glob("*/*.jpg")))
  roses = list(data_dir.glob("roses/*"))
  tulips = list(data_dir.glob("tulips/*"))
  print("total: " + str(image_count))
  print("roses: " + str(len(roses)))
  print("tulips: " + str(len(tulips)))

  return data_dir


# 사진 전처리
# arguments: date_dir (텐서플로우 파일 객체?), batch_size (Int), img_height (Int), img_width (Int)
# return: train_ds (Dataset), val_ds (Dataset), class_name (list of String)
def preprocess(data_dir, batch_size, img_height, img_width):  
  # 주어진 파일 객체에서 이미지들을 전처리하여 training, validation set을 얻는다.
  train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size
  )
  val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size
  )
  # 분류되는 클래스의 개수를 구한다.
  class_names = train_ds.class_names
  return train_ds, val_ds, class_names


# 데이터 시각화
# arguments: train_ds (Dataset)
# return: ()
def visualize(train_ds):
  # 처음 9개의 사진 시각화
  plt.figure(figsize=(10, 10))  
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
  plt.show()


# 모델 생성
# arguments: num_classes (Int), batch_size (Int), img_height (Int), img_width (Int)
# return: model (Model), history (History)
def create_model(num_classes, batch_size, img_height, img_width):
  # 여기서는 모델 첫 계층에 표준화를 진행한다.
  model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])
  # optimizer, loss, metrics를 설정하여 컴파일하고 정보를 출력하고 반환한다.
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  return model


# 모델 학습 및 학습 기록 반환
# arguments: model (Model), train (Dataset), val (Dataset), epochs (Int)
# return: history (History)
def train_model(model, train, val, epochs):
  history = model.fit(train, validation_data=val, epochs=epochs)
  return history


# 훈련 결과 시각화
# arguments: history (History), epochs (Int)
# return: ()
def visualize_history(history, epochs):
  # training set, validation set의 정확도를 구한다.
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  # training set, validation set의 손실를 구한다.
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  # 정확도 그래프
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  # 손실 그래프
  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


# 메인 함수
if __name__ == "__main__":
  # 몇몇 파라미터를 정한다.
  batch_size = 32
  img_width = 180
  img_height = 180
  epochs = 10

  # 원격 저장소에서 데이터 로드
  data_dir = load_data()
  # 전처리하여 training set, validation set 생성
  train, val, class_names = preprocess(data_dir, batch_size, img_height, img_width)
  # 처음 9개의 사진 시각화
  visualize(train)
  # 모델 생성
  model = create_model(len(class_names), batch_size, img_height, img_width)
  # 모델 학습
  history = train_model(model, train, val, epochs)
  # 훈련 결과 시각화
  visualize_history(history, epochs)
