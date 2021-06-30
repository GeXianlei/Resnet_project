from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import itertools
import os
im_height = 512
im_width = 512
batch_size = 8
epochs = 10
classes = 2


if not os.path.exists("save_weights"):
    os.makedirs("save_weights")


image_path = "./targets/"
train_dir = image_path + "train"
validation_dir = image_path + "val"
test_dir = image_path + "test"
train_image_generator = ImageDataGenerator( rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')

total_train = train_data_gen.n

val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')

total_val = val_data_gen.n

test_data_gen = test_image_generator.flow_from_directory(directory=test_dir,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')

total_test = test_data_gen.n
# 只要库中有就可以使用
covn_base = tf.keras.applications.ResNet50(weights='imagenet', include_top = False,input_shape=(im_height,im_width,3))
# 冻结基础层
covn_base.trainable = False

model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(classes, activation='softmax'))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=2,
                              mode='auto',
                              verbose=1)


checkpoint = ModelCheckpoint(filepath='./save_weights/DenseNet201.ckpt',
                             monitor='val_acc',
                             save_weights_only=False,
                             save_best_only=True,
                             mode='auto',
                             period=1)

history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    callbacks=[checkpoint, reduce_lr])

# 保存模型
model.save_weights('./save_weights/demo.h5',save_format='h5')

history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["acc"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_acc"]

#损失值
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig("./loss.png")

# 准确率
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig("./acc.png")
plt.show()


# 评估模型的代码
scores = model.evaluate(test_data_gen, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm)) #计算准确率
    misclass = 1 - accuracy #计算错误率
    if cmap is None:
        cmap = plt.get_cmap('Blues') #颜色设置成蓝色
    plt.figure(figsize=(10, 8)) #设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #显示图片
    plt.title(title) #显示标题
    plt.colorbar() #绘制颜色条

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45) #x坐标标签旋转45度
        plt.yticks(tick_marks, target_names) #y坐标

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm,2) #对数字保留两位小数
        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize: #标准化
            plt.text(j, i, "{:0.2f}".format(cm[i, j]), #保留两位小数
                     horizontalalignment="center",  #数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  #设置字体颜色
        else:  #非标准化
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",  #数字在方框中间
                     color="white" if cm[i, j] > thresh else "black") #设置字体颜色

    plt.tight_layout() #自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label') #y方向上的标签
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass)) #x方向上的标签
    plt.savefig("./confu.png")
    plt.show() #显示图片

labels = ['Cancerous','NonCancerous']

# 预测验证集数据整体准确率
Y_pred = model.predict_generator(test_data_gen, total_test // batch_size + 1)
# 将预测的结果转化为one hit向量
Y_pred_classes = np.argmax(Y_pred, axis = 1)
# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true = test_data_gen.classes,y_pred = Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)

# 测试模型：输入单张图像输出预测结果
# #获取数据集的类别编码
# class_indices = train_data_gen.class_indices
# #将编码和对应的类别存入字典
# inverse_dict = dict((val, key) for key, val in class_indices.items())
# #加载测试图片
# img = Image.open("../input/chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg")
# # 将图片resize到224x224大小
# img = img.resize((im_width, im_height))
# #将灰度图转化为RGB模式
# img = img.convert("RGB")
# # 归一化
# img1 = np.array(img) / 255.
# # 将图片增加一个维度，目的是匹配网络模型
# img1 = (np.expand_dims(img1, 0))
# #将预测结果转化为概率值
# result = np.squeeze(model.predict(img1))
# predict_class = np.argmax(result)
# #print(inverse_dict[int(predict_class)],result[predict_class])
# #将预测的结果打印在图片上面
# plt.title([inverse_dict[int(predict_class)],result[predict_class]])
# #显示图片
# plt.imshow(img)
# """
