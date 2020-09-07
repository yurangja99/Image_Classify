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

# 몇몇 파라미터 값 설정
batch_size = 32
img_height = 180
img_width = 180
epochs = 10


# 사진 데이터 캐싱

