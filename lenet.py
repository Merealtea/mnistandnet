from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import matplotlib.pyplot as plt



# Example Sigmoid
# 这个类中包含了 forward 和backward函数
class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))


## 在原 LeNet-5上进行少许修改后的 网路结构

def forward(x,kernel): #实现卷积的后向传播的卷积
        N,C,H,W = x.shape
        B,cc,hh,ww = kernel.shape
        x = x.astype(np.float)
        kernel = kernel.astype(np.float)
        L = 8
        [outN ,outC,outH,outW] = [N,B,H - hh +1,W - ww + 1]

        x = np.lib.stride_tricks.as_strided(x,shape = (N,1,outH,outW,cc,hh,ww),strides = (W*H*C*L,W*H*L,W*L,L,W*H*L,W*L,L))
        out = np.einsum("nchwpqk,bpqk->nbhw",x,kernel)
        return out


class Conv(object):
    def __init__(self,activation):
        self.activation = activation

    def forward(self,x,kernel):
        N,C,H,W = x.shape
        B,cc,hh,ww = kernel.shape
        x = x.astype(np.float64)
        kernel = kernel.astype(np.float64)
        self.input = x
        self.kernel = kernel
        L = 8
        [outN ,outC,outH,outW] = [N,B,H - hh +1,W - ww + 1]

        x = np.lib.stride_tricks.as_strided(x,shape = (N,1,outH,outW,cc,hh,ww),strides = (W*H*C*L,W*H*L,W*L,L,W*H*L,W*L,L))
        self.out = self.activation.forward(np.einsum("nchwpqk,bpqk->nbhw",x,kernel))
        return self.out

    
    def backward(self,error, lr):
        o_delte = error * self.activation.backward(self.out)
        B0,C0,H0,W0 = o_delte.shape
        B,C,hh,ww = self.kernel.shape
        x = self.input
        x = np.transpose(x,(1,0,2,3))
        t = np.transpose(o_delte,(1,0,2,3))
        w_delte = np.transpose(forward(x,t),(1,0,2,3))#计算卷积核的delta

        input_delte = np.zeros(shape = self.input.shape)
        p = np.transpose(self.kernel,(1,0,2,3))
        for i in range (0,hh):
            for j in range (0,ww):
                ker = p[:,:,i,j]
                ker = np.expand_dims(ker,axis = 2)
                ker = np.expand_dims(ker,axis = 3)
                ker = np.broadcast_to(ker,shape = (C,B,H0,W0))
                input_delte[:,:,i:i + H0,j:j+W0] += np.einsum("kglc,pglc->kplc",o_delte,ker)#计算输入的delta

        self.kernel -= lr * w_delte
        return input_delte, self.kernel

class relu(object):
    def __init__(self):
        pass

    def forward(self,x): 
        return np.maximum(x,0)

    def backward(self,z):
        return np.where(z == 0, 0 ,1)


class Avgpooling(object):
    def __init__(self):
        pass

    def forward(self,x):
        N,C,H,W = x.shape
        x = 0.25 * x.reshape((N*C*H//2,2,W))
        out = np.zeros((N*C*H//2,W//2))
        for i in range(0,W,2):
            out[:,i//2] = np.sum(x[:,:,i:i+2],axis = (1,2))
        return out.reshape(N,C,H//2,W//2)

    def backward(self,error):
        error = error.repeat(2,axis = 2)
        error = error.repeat(2,axis = 3)
        return error * 0.25

class FC(object):
    def __init__(self,activation):
        self.activation = activation

    def forward(self,x,W):
        self.W = W
        self.input = x
        temp = np.dot(x,self.W)
        self.output = self.activation.forward(temp)
        return self.output

    def backward(self,error,lr):
        O_deltar = error * self.activation.backward(self.output)
        W_deltar = np.matrix(self.input).T.dot(O_deltar)
        input_deltar = O_deltar.dot(self.W.T)
        self.W -= lr * W_deltar
        return input_deltar,self.W


class softmax(object):
    def __init__(self):
        pass

    def forward(self, x):
        M = np.max(x,axis = 1)
        M = np.expand_dims(M,axis = 1)
        m = np.min(x,axis = 1,out = None,keepdims = 1)
        x =  x - M 
        exp_x = np.exp(x)
        self.output = exp_x/np.sum(exp_x,axis = 1,keepdims = 1)
        return self.output

    def backward(self, labels):   #交叉熵loss函数反传求导
        return self.output - labels

class LeNet(object):
    def __init__(self):
        
        self.activation = relu()
        self.conv1 = Conv(self.activation) #6个1通道5*5的矩阵
        self.conv2 = Conv( self.activation)#16个6通道5*5的矩阵

        self.pooling = Avgpooling()
        self.FC1 = FC(self.activation)
        self.FC2 = FC(self.activation)
        self.FC3 = FC(self.activation)

        self.classifier = softmax()

        self._intweight()

        print("initialize")

    def _intweight(self):
        self.kernel1 = np.random.uniform(np.sqrt(6/(5*5*1+6*5*5)),-np.sqrt(6/(5*5*1+6*5*5)),(6,1,5,5),)
        self.kernel2 = np.random.uniform(np.sqrt(6/(6*5*5 + 16 * 5 * 5)),-np.sqrt(6/(6*5*5 + 16 * 5 * 5)),(16,6,5,5))
        self.Fully1 = np.random.uniform(np.sqrt(6/(256 + 128)),-np.sqrt(6/(256 + 128)),(256,128))
        self.Fully2 = np.random.uniform(np.sqrt(6 /(128 + 64)),-np.sqrt(6 /(128 + 64)),(128,64))
        self.Fully3 = np.random.uniform(np.sqrt(6/(64 + 10)),-np.sqrt(6/(64 + 10)),(64,10))
        
    def forward(self, x):

        if x.ndim == 3:
            x = np.expand_dims(x,axis = 1)
 
        B,C,H,W = x.shape
        self.B = B
        self.input = x
        Conv1 = self.conv1.forward(x,self.kernel1)
        Avg1 = self.pooling.forward(Conv1)
        Conv2 = self.conv2.forward(Avg1,self.kernel2)
        Avg2 = self.pooling.forward(Conv2)   #downsize
        AB ,AC,AH,AW = Avg2.shape
        flatten = Avg2.reshape((AB,AC * AH * AW))  #flatten

        FullyC1 = self.FC1.forward(flatten,self.Fully1)#FC
        
        FullyC2 = self.FC2.forward(FullyC1,self.Fully2)
        
        FullyC3 = self.FC3.forward(FullyC2,self.Fully3)
        
        
        self.out = self.classifier.forward(FullyC3)

        return self.out

    def backward(self, labels, lr):

        delte1 = self.classifier.backward(labels)#softmax

        delte2 ,self.Fully3 = self.FC3.backward(delte1,lr)     
        delte3, self.Fully2 = self.FC2.backward(delte2,lr)
        delte4, self.Fully1 = self.FC1.backward(delte3,lr)#FC

        delte4 = delte4.reshape((self.B,16,4,4))#reshape

        delte5 = self.pooling.backward(delte4)
        delte6, self.kernel2 = self.conv2.backward(delte5,lr)#Conv2
        delte7 = self.pooling.backward(delte6)
        delte8 ,self.kernel1 = self.conv1.backward(delte7,lr)#Conv1
    
    def evaluate(self, x, labels,batch_size):

        N,H,W = x.shape
        N,L = labels.shape
        batch_number = N // batch_size if N % batch_size == 0 else N // batch_size + 1
        percent = 0
        loss_sum = 0
        for i in range(batch_number):
            image = x[i*batch_size:i*batch_size + batch_size,:,:]
            label = labels[i*batch_size:i*batch_size + batch_size,:]
            test = self.forward(image)
            A,B = label.shape
            temp = np.argmax(test,axis = 1)
            test = np.eye(A,B)[temp] 
            right = 0
            for i in range(0,A):
                if (test[i,:] == label[i,:]).all():
                    right += 1 
            loss = self.computer_loss(test,label)
            loss_sum += loss
            percent += right/A
        percent = percent / batch_number
        loss_sum = loss_sum/ batch_number
        return percent, loss_sum

    def computer_loss(self, pre, labels):
        loss = np.sum((pre - labels)**2) / labels.shape[0]#交叉熵
        return loss

    def decay(self,learning_rate,times):#调整学习率下降
        return np.exp(-0.01 * times) * learning_rate

    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 10,
        batch_size = 16,
        lr = 1
    ):
        sum_time = 0
        accuracies = []
        S, H, W = train_image.shape
        N , L = train_label.shape
        c = 0 

        train_loss = []
        test_loss = []
        test_acc = []
        train_acc = []

        for epoch in range(epoches):

            ## 可选操作，数据增强
            #train_image = self.data_augmentation(train_image)
            ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应


            
            temp1 = train_image[:S//2,:,:]
            temp2 = train_image[S//2:,:,:]
            t1 = train_label[:S//2,:]
            t2 = train_label[S//2:,:]
            n1 = temp1.shape[0]
            n2 = temp2.shape[0]

            for i in range(S):
                if i % 2 == 1 and i//2 <= n1:
                    train_image[i,:,:] = temp1[i//2,:]
                    train_label[i,:] = t1[i//2,:]
                elif i//2<= n2:
                    train_image[i,:,:] = temp2[i//2,:]
                    train_label[i,:] = t2[i//2,:]      #打乱顺序
            

            batch_number = np.ceil(S / batch_size).astype(np.int32)

            batch_images = []
            batch_labels = []

            for i in range(0,batch_number):
                images = train_image[i * batch_size : (i+1)*batch_size,:,:]
                labels = train_label[i * batch_size : (i+1)*batch_size,:]
                batch_images.insert(i, images)
                batch_labels.insert(i, labels)

            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                pre = self.forward(imgs)
                self.backward(labels,lr)

            accu ,loss_sum =  self.evaluate(train_image,train_label,batch_size)
            train_loss.append(loss_sum)
            train_acc.append(accu)
            
            accu , loss_sum = self.evaluate(test_image,test_label,batch_size)
            test_loss.append(loss_sum)
            test_acc.append(accu)

            c += 1
            lr = self.decay(lr, c)
            duration = time.time() - last
            sum_time += duration

            
            if epoch % 2 == 0:
                accuracy = self.evaluate(test_image, test_label,batch_size)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)
            

        plt.plot(train_loss,'-o',label = 'train loss')
        plt.plot(test_loss,'-x', label='testing loss') 
        plt.legend()
        plt.figure()
        plt.plot(train_acc,'-o',label = 'train accuracy')
        plt.plot(test_acc, '-x', label='testing accuracy') 
        plt.legend()

        avg_time = sum_time / epoches
        return avg_time, accuracies
