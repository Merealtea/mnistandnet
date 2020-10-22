from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from lenet import LeNet
import csv

def normalize_image(images):   #图像归一化到-1到1
    N,H,W = images.shape
    image = np.zeros((N,H,W))
    for i in range(10):
        p = images[i * N//10: i*N//10 + N//10,:,:]
        p = p / 255
        p = p * 2 - 1
        image[i * N //10: i * N//10 + N//10:,:,:] = p
    return image

def one_hot_labels(labels):    #label转化为one hot vector
    return np.eye(np.max(labels)+1)[labels] 


def main():
    with open('train.csv','rt') as f1:
       train = np.loadtxt(f1,dtype = int, delimiter = ',',converters = None, skiprows = 1)
       labels = train[:,0]
       img = np.reshape(train[:,1:],(-1,28,28))   #载入图片数据

    x_train = normalize_image(img[:-3000,:,:])
    x_test = normalize_image(img[-3000:,:,:])
    y_train = one_hot_labels(labels[:-3000])
    y_test = one_hot_labels(labels[-3000:])   #图片和标签预处理

    net = LeNet()
    batch_size = 32
    net.fit(x_train, y_train, x_test, y_test, epoches=14, batch_size=32, lr = 0.0016)#开始训练模型

    accu = net.evaluate(x_test, labels=y_test,batch_size = 32)
    print("final accuracy {}".format(accu))

    with open('test.csv','rt') as f2:     #载入测试集并进行预处理
          temp = np.loadtxt(f2,dtype = int, delimiter = ',',converters = None, skiprows = 1)
          ans = normalize_image(np.reshape(temp,(-1,28,28)))

    N = ans.shape[0]       
    res = np.zeros(N)
    num = np.ceil(N / batch_size).astype(np.int32)
    vec = np.array([0,1,2,3,4,5,6,7,8,9])

    for i in range(num):   #测试集预测
        lab = net.forward(ans[i*batch_size: (i+1) * batch_size,:,:])
        A, B = lab.shape
        loc = np.argmax(lab,axis = 1)
        label = np.sum(np.eye(10)[loc] * vec,axis = 1)
        res[i*batch_size: (i+1)*batch_size] = label

    res = res.astype(np.str)
    with open('try.csv','w',newline = '') as f3:    #预测结果导入csv文件
        writer = csv.writer(f3)
        writer.writerows(res)

if __name__ == "__main__":
    main()
