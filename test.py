from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import itertools
import os
im_height = 224
im_width = 224
batch_size = 16
epochs = 50
CLASS = 2


if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

image_path = "./targets/"
train_dir = image_path + "train"

train_image_generator = ImageDataGenerator(rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           rescale=1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest'
                                          )

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')

model = tf.keras.Sequential()
# 只要库中有就可以使用
covn_base = tf.keras.applications.DenseNet201(weights=None, include_top = False,input_shape=(im_height,im_width,3))
# 冻结基础层
covn_base.trainable = False

model.add(covn_base)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(CLASS, activation='softmax'))
model.summary()

model.load_weights('./save_weights/model.h5')

# 测试模型：输入单zzzzz张图像输出预测结果
#获取数据集的类别编码
class_indices = train_data_gen.class_indices
#将编码和对应的类别存入字典
inverse_dict = dict((val, key) for key, val in class_indices.items())
#加载测试图片
img = Image.open("./5.jpg")
# 将图片resize到224x224大小
img = img.resize((im_width, im_height))
#将灰度图转化为RGB模式
img = img.convert("RGB")
# 归一化
img1 = np.array(img) / 255.
# 将图片增加一个维度，目的是匹配网络模型
img1 = (np.expand_dims(img1, 0))
#将预测结果转化为概率值
result = np.squeeze(model.predict(img1))
predict_class = np.argmax(result)
#print(inverse_dict[int(predict_class)],result[predict_class])
#将预测的结果打印在图片上面
plt.title([inverse_dict[int(predict_class)],result[predict_class]])
#显示图片
plt.imshow(img)
plt.savefig('result5.png')