# -*-Otto Group 商品识别-*-
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
import tensorflow as tf
from keras import backend as K

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# ------ 读取数据 ------ #
train_data = pd.read_csv(open("train.csv"))  # 读取train.csv的数据
test_data = pd.read_csv(open("test.csv"))    # 读取test.csv的数据
##print(train_data.info())                   # 显示train.csv数据的信息
train_y_raw = train_data["target"]           # 把商品类别的数据存入train_y_raw，49502个商品的类别
##print(train_y_raw)
x_label = []
for i in range(1, 94):
    x_label.append("feat_%s"  %(i))          # x_label存入feat数据(feat_1,feat_2……feat_93)
train_x = np.array(train_data[x_label])      # train_x存入train.csv文件中所有商品的feat信息
test_x = np.array(test_data[x_label])        # test_x存入test.csv文件中所有商品的feat信息



# ------ 将train_y的数据转换成one_hot向量(9维) ------ #
train_y = np.zeros([len(train_y_raw), 9])    # 构建train_y矩阵(49502*9)
for i in range(len(train_y_raw)):
    lable_data = int(train_y_raw[i][-1])     # lable_data存入了49502个商品的类别号
    train_y[i, lable_data-1] = 1             # train_y存入class的one_hot向量(class7=000000100)
##print(train_x.shape, train_y.shape, test_x.shape)# (49502, 93) (49502, 9) (12376, 93)



# ------ 构建模型与模型训练，93-128-64-32-16-9神经网络结构 ------ #
model = Sequential()                         # 序贯（Sequential）模型(多个网络层的线性堆叠)
model.add(Dense(128, input_shape=(93,), activation="relu"))  
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(9))
model.add(Activation('softmax'))             # 前几层用relu激活函数，最后一层使用softmax激活函数
#model.summary()
model.compile(loss='mean_squared_logarithmic_error',
              optimizer='adadelta', metrics=['accuracy'])
model.fit(x = train_x, y = train_y, batch_size = 2048, nb_epoch = 250)  # 训练模型，分批迭代样本的数据



# ------ 预测答案 ------ #
test_y = model.predict(test_x)
print(test_y.shape)
answer = pd.read_csv(open("sampleSubmission.csv"))
class_list = ["Class_1", "Class_2", "Class_3", "Class_4",
              "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"]
answer[class_list] = answer[class_list].astype(float)



# ------ 答案存入submission.csv文件 ------ #
j = 0
for class_name in class_list:
    answer[class_name] = test_y[:, j]
    j += 1
answer.to_csv("submission.csv", index=False) # 不要保存引索列
