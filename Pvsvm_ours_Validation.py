#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import gmpy2
from pypbc import *
import math
import pandas as pd
import csv
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def get_minist_data():
    # 数据加载
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_image01 = train_images[np.where(train_labels <= 1)[0]]
    train_label01 = train_labels[np.where(train_labels <= 1)[0]]

    train_image01_flat = train_image01.reshape(train_image01.shape[0], 784)
    print('样本数', train_image01.shape[0])
    # PCA降维
    # 创建一个模型叫 model
    pca = PCA(n_components=20)  # n_components
    # #train_image01_flat.shape
    reduced_train_image01_flat = pca.fit_transform(train_image01_flat)
    print(reduced_train_image01_flat[0])
    one = np.zeros(len(reduced_train_image01_flat)) + 1
    reduced_train_image01_flat = np.column_stack((reduced_train_image01_flat, one))  # 扩展维度
    print(reduced_train_image01_flat[0])
    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(reduced_train_image01_flat, train_label01, test_size=0.3,
                                                        shuffle=True, random_state=3)
    # #获取初始数据
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    X_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return  X_train, X_test, y_train, y_test

def get_iris_data():# 加载iris数据集
    iris = datasets.load_iris()
    # iris.data大小为150*4,代表4种特征，这里可以只提取后两类特征 x = iris.data[:, [2, 3]]
    x = iris.data
    y = iris.target
    first_id = list(y).index(2)
    #x = np.trunc(x[:first_id - 1])#取整
    x = x[:first_id - 1]  # 取整
    one = np.zeros(len(x))+1
    x = np.column_stack((x,one))#扩展维度
    #print(x, type(x))
    y = y[:first_id - 1]#观察数据，选择label：0，1
    y = -(y*(-2)+1)#将label:0转换为label:1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=2)
    x_train = x_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = x_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return  x_train, x_test, y_train, y_test

def get_heart_data():
    # 导入数据
    df = pd.read_csv('UCI Heart Disease Dataset.csv')
    df['sex'][df['sex'] == 1] = 'male'
    df['cp'][df['cp'] == 0] = 'typical angina'
    df['cp'][df['cp'] == 1] = 'atypical angina'
    df['cp'][df['cp'] == 2] = 'non-anginal pain'
    df['cp'][df['cp'] == 3] = 'asymptomatic'

    df['fbs'][df['fbs'] == 0] = 'lower than 120mg/ml'
    df['fbs'][df['fbs'] == 1] = 'greater than 120mg ml'

    df['restecg'][df['restecg'] == 0] = 'normal'
    df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'
    df['restecg'][df['restecg'] == 1] = 'left ventricular hyper trophy'

    df['exang'][df['exang'] == 0] = 'no'
    df['exang'][df['exang'] == 1] = 'yes'

    df['slope'][df['slope'] == 0] = 'upsloping'
    df['slope'][df['slope'] == 1] = 'flat'
    df['slope'][df['slope'] == 1] = 'downsloping'

    df['thal'][df['thal'] == 0] = 'unknown'
    df['thal'][df['thal'] == 1] = 'normal'
    df['thal'][df['thal'] == 1] = 'fixed defect'
    df['thal'][df['thal'] == 1] = 'reversable defect'

    # 将离散的定类和定序特征列转为One-Hot独热编码
    # 将定类数据扩展为特征
    df = pd.get_dummies(df)
    # df.head(3)

    y = df.target.values
    y = (y * (2) - 1)  # 将label:0转换为label:-1

    X = df.drop(['target'], axis=1)
    X = np.array(X)
    # pca = PCA()
    # pca.fit(X)
    # print('降维前',X.shape)
    # X = pca.transform(X)  # 用它来降低维度
    # print('降维后',X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)  # 13
    # 数据标准化处理
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one))  # 扩展维度
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one))  # 扩展维度
    x_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return x_train, x_test, y_train, y_test

def get_breast_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 1

    sc_X = StandardScaler()# 数据标准化处理
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one))  # 扩展维度
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one))  # 扩展维度
    x_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return x_train, x_test, y_train, y_test

def get_Ionosphere_data():
    data_filename = "ionosphere.data"
    X = np.zeros((351, 34), dtype='float')
    y = np.zeros((351,), dtype='bool')

    with open(data_filename, 'r') as input_file:
        reader = csv.reader(input_file)
        # print(reader)  # csv.reader类型
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            # Set the appropriate row in our dataset
            X[i] = data
            # 将“g”记为1，将“b”记为0。
            y[i] = bool(row[-1] == 'g')
    y = 2 * y.astype(int) - 1
    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one))  # 扩展维度
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one))  # 扩展维度
    x_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return x_train, x_test, y_train, y_test

def get_prime(rs):
    p = gmpy2.mpz_urandomb(rs, 512)
    while not gmpy2.is_prime(p) & gmpy2.is_prime((p-1)//2):
        p = p + 1
    return p

def L(x, n):
    return (x - 1) // n

def pdec(c,lmd1,lmd2,pk):
    Partial_c_Wb_1 = partialdec(pk, lmd1, c)
    Partial_c_Wb_2 = partialdec(pk, lmd2, c)
    Partial_c_Wb = partialcombine(pk, Partial_c_Wb_1, Partial_c_Wb_2) / (10 ** 12)
    return(Partial_c_Wb.astype(np.float64))

def keygen():
    rs = gmpy2.random_state(int(time.time()))
    p = get_prime(rs)
    q = get_prime(rs)

    n = p * q

    lmd = (p - 1) * (q - 1) // 2
    a = random.randint(1,n**2)
    g = -gmpy2.powmod(a,2*n,n**2)

    pk = [n, g]

    sk = lmd
    return pk, sk

def encipher(pk, plaintext):
    m = plaintext
    n, g = pk
    r = random.randint(1, n / 4)
    if m<=0:
        m = -1*m
    c = gmpy2.powmod(g,r,n**2) * (1 + m * n) % (n**2)
    return c

def decipher(pk,sk,c):
    n, g = pk
    lmd = sk
    m = L(gmpy2.powmod(c,lmd,n**2),n)
    m = m / lmd % n
    return m

def secrekeySplit (pk,sk):
    lmd = sk
    n,g = pk
    s = (lmd * gmpy2.invert(lmd,n**2)) % (lmd * (n ** 2))#模逆运算

    lmd1 = random.randint(1, s)
    lmd2 = s - lmd1

    return lmd1, lmd2

def partialD(pk,sk,c):
    [n,g] = pk
    m = gmpy2.powmod(c,sk,n**2)
    return m

def partialR(pk,c1,c2):
    [n,g] = pk
    a = c1 * c2 % (n ** 2)
    m = L(a,n)
    return m

def partialdec(pk,sk,c):
    m = []
    for i in range(c.shape[0]) :
        for j in range (c.shape[1]):
            ciphertext = int(c[i][j])
            x = int(partialD(pk, sk, ciphertext))
            m.append(x)
    m = np.array(m).reshape(c.shape[0],c.shape[1])
    return m

def partialcombine(pk,c1,c2):
    m = []
    for i in range(c1.shape[0]) :
        for j in range (c1.shape[1]):
            x = int(partialR(pk,c1[i][j],c2[i][j]))
            m.append(x)
    m = np.array(m).reshape(c1.shape[0],c1.shape[1])
    return m

def enc(m,pk):
    c = []
    for i in range(m.shape[0]) :
        for j in range (m.shape[1]):
            plaintext = int(m[i][j])
            x = int(encipher(pk, plaintext))
            c.append(x)
    c = np.array(c).reshape(m.shape[0],m.shape[1])
    return c

def dec(c,pk,sk):
    m = []
    for i in range(c.shape[0]) :
        for j in range (c.shape[1]):
            ciphertext = int(c[i][j])
            x = int(decipher(pk, sk, ciphertext))
            m.append(x)
    m = np.array(m).reshape(c.shape[0],c.shape[1])
    return m

def V_correctness_Ci(plain_C,g__wj,h__xijyi):#批量验证
    q_list = np.random.randint(1, 10, size=plain_C.shape[1])# p -> inf
    e1 = 1
    sum = 0
    #gmpy2.mpz(sum)
    for i in range(q_list.shape[0]):
        sum = sum +  int(q_list[i])*plain_C[0][i]
    e1=g**int(sum)#<class 'pypbc.Element'> 024F3E49
    h__q_xijyi = [1] * len(h__xijyi[0])#5
    for j in  range(len(h__xijyi[0])):#5
        temp_first = h__xijyi[0][j] ** int(q_list[j])
        for i in range(1,len(h__xijyi)):#79
            temp_first = temp_first * h__xijyi[i][j] ** int(q_list[j])#element  int gmpy2.powmod(int(h__xijyi[0][j]), int(q_list[j]), N ** 2) % N**2#
        h__q_xijyi[j] = temp_first
    eba = pairing.apply(g__wj[0],h__q_xijyi[0])
    for j in range(1,len(g__wj)):
        eba = eba * pairing.apply(g__wj[j],h__q_xijyi[j])
    return pairing.apply(e1, h)==eba

def V_correctness_Ci_V1(plain_C,g__wj,h__xijyi):#非批量验证
    ver_list = []
    for i in range(plain_C.shape[1]):
        e1 = pairing.apply(g ** int(plain_C[0][i]), h)#heart (0,237)
        temp_first = pairing.apply(g__wj[0], h__xijyi[i][0])
        for j in range(len(h__xijyi[0])):  # 5
            temp_first = temp_first * pairing.apply(g__wj[j], h__xijyi[i][j])

        ver_list.append( temp_first == e1)
    return all(ver_list)

# print('双线性 累乘验证',pairing.apply(g**2,h**2),pairing.apply(g,h)*pairing.apply(g,h))

#***********************************************************************************************************************
time_start = time.perf_counter()
# #获取初始数据
#X_train, X_test, y_train, y_test = get_iris_data()#鸢尾花
#X_train, X_test, y_train, y_test = get_heart_data()#心脏病
#X_train, X_test, y_train, y_test = get_breast_data()#乳腺癌
#X_train, X_test, y_train, y_test = get_Ionosphere_data()#离子
X_train, X_test, y_train, y_test = get_minist_data()#minist

'''Initialization_Step1_TTP'''
# #Key_Gen
pk, sk = keygen()#sk 随机性的问题
lmd1, lmd2 = secrekeySplit(pk,sk)
PK = pk[1]
N = pk[0]

#***********************************************************************************************************************
#构建双线性映射 pypbc
# #根据安全参数进行实例化 https://www.freesion.com/article/19881431966/
# qbits=512
# rbits=160
# params = Parameters(qbits=qbits, rbits=rbits)   #参数初始化  应选择a类曲线以保证G1 X G1 = G2

#根据设置群的阶数来生成曲线
q_1 = get_random_prime(20)  # 生成一个长度为30的随机素数
q_2 = get_random_prime(20)
p = q_1 * q_2 # k
#使用的是pbc中的a1_param参数
params = Parameters( n=p )
pairing = Pairing(params)  # 根据参数实例化 双线性对
#从群中取一个随机数，并初始化一个元素，一般是取g的，也就是生成元。
g = Element.random(pairing, G1)  # g h 是G1的一个生成元
h = Element.random(pairing, G1)  #type(G1):<class 'int'>

# '''双线性对运算如下:# e(g,g)
#e = pairing.apply(g,g)
#
# 对群进行运算一般使用Element，而不是直接在数值上进行运算。（当然直接运算也可以）。
# 其中pairing代表我们初始化的双线性对，G2代表返回值的类型，value=就是值等于多少，G2中的元素做底数，Zr中的元素做指数，
# 其实也能使用b = g ** c是同样的效果，但下面这样写更加工整，看着更明白，减少出错。
# b = Element( pairing, G2, value = g ** c )   # b = g^c
# '''
#global gp
gp = [pairing,p,G1,G2,g,h]
#***********************************************************************************************************************
'''Initialization_Step2'''
def data_owner_init(gp):
    DO_a=[]
    DO_pk=[]
    DO_hv=[]
    for i in range(len(y_train)):
        ai = Element.random(pairing,Zr)#Zp element
        DO_a.append(ai)
        DO_pk.append((gp[4] ** ai,gp[5] ** ai))#g h
        DO_hv.append(gp[5] ** (ai**(-1)))
    return  DO_a,DO_pk,DO_hv

def model_request_init(gp):
    ar = Element.random(pairing,Zr)#sk
    MR_a = ar
    MR_pk = (gp[4] ** ar,gp[5] ** ar)
    return MR_a,MR_pk

DO_a,DO_pk,DO_hv = data_owner_init(gp)
MR_a,MR_pk = model_request_init(gp)
#***********************************************************************************************************************
'''Data Submission_Step1:Model_Request'''
MR_DS_start = time.perf_counter()
#随机初始化权重
#print(np.random.normal(loc=0, scale=0.01, size=(5,2)).astype(np.float64))
#W = np.random.randn(len(X_train[0]))#  [round(x) for x in np.random.normal(0,2,4)]
np.random.seed(1)
W = np.random.normal( loc=0, scale=0.01, size=(len(X_train[0]),) ).astype(np.float64)
#W = np.array([1,1,1,1,1]).astype(np.float64)#给定固定值
W[-1]=np.float64(1.0)
omega = np.sum(W)
#print('omega',omega)
#W=W*(10**8))#f放大取整

#随机初始化扰动值
r = np.random.normal(loc=0, scale=0.01, size=(len(X_train[0]),)).astype(np.float64)
r[-1]=np.float64(0)
#r = r*(10**8)
#r=[round(i) for i in r]#取整

#添加扰动值
Wb =  np.sum([W, r], axis=0)

K_list = np.random.randint(0,p,size=(2,len(X_train[0])))#Zp范围内？****************************    h?
K1,K2= K_list[0].astype(int),K_list[1].astype(int)
K3 = np.random.randint(0,p,size=(1))

#print('验证双线性映射', pairing.apply(g ** K1, h ** K2)== pairing.apply(g, h) ** (K1 * K2))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
U_wj = [ (gp[-1]**int(x)) for x in K1]#h  list
U_rj = [gp[-1]**int(x) for x in K2]
U_W = [gp[-1]**int(K3[0])]# omega list

#tag
# print(MR_a,int(MR_a),Wb[0],K1,p)
# print(type(MR_a),type(Wb[0]),type(K1),type(p))
emc_wj = [(int(MR_a)*(Wb[j]+K1[j]))%p for j in range(len(Wb))]   #print(1**0x11095A059DA86A==1**8908973441865092) true(作为指数计算时十六进制和十进制无区别)
#omg = [w**2 for w in W]
#emc_W = [(int(MR_a)*(omg[j]+K2[j]))%p for j in range(len(omg))]#
emc_rj = [(int(MR_a)*(r[j]+K2[j]))%p for j in range(len(r))]#
emc_W  = [(int(MR_a)*(omega+K3[0]))%p ]

# the model requester : helper values
g__wj = [gp[4]**int(wj*10**6) for wj in W]#   W*10**6   实际放大应到10**8
h__arai = [(gp[5]**(MR_a))**(DO_ai**(-1)) for DO_ai in DO_a]#注意数据类型，是否会损失精度
# the model requester : encrypts the model weights
Wb = np.array(Wb).reshape(Wb.shape[0],1)#8->int
c_Wb = enc(Wb*(10**6), pk)#为便于计算书写，可列表转换为数组类型  shape(5,1)

#send:MR_SA
MR_TO_SA=[ (g__wj,Wb,(U_wj,emc_wj)) , (U_rj,emc_rj) , h__arai , (U_W,emc_W) ]
#未完全按照文中顺序
#send:MR_SB
MR_TO_SB=[( g__wj , r , (U_rj,emc_rj) , (U_wj,emc_wj) , c_Wb ),h__arai]
#***********************************************************************************************************************
MR_DS_end = time.perf_counter()

'''Data Submission_Step2:Data owner'''
DO_DS_start = time.perf_counter()
#extends his/her training data，在数据初始化时（数据划分前）已进行该操作

# multiplies the training data with its corresponding lables
xi_yi=np.array([X_train[i]*y_train[i] for i in range(len(y_train))])

#encrypt
c_xi_yi = enc(xi_yi*(10**6), pk)#(79, 5) shape
k_ij = []
U_xij_yi=[]
emc_xij_yi=[]

for x in range(c_xi_yi.shape[0]):
    u=[]
    emc=[]
    for y in range(c_xi_yi.shape[1]):
        kij = int(Element.random(pairing,Zr))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        k_ij.append(kij)
        u.append(gp[5] ** kij)  # **************************数据类型element int
        emc.append((int(DO_a[x]) * (xi_yi[x][y] + kij)) % p)#先将 DO_a[x] （从Zp中选出的）十六进制转十进制
    U_xij_yi.append(u)
    emc_xij_yi.append(emc)

k_ij= np.array(k_ij).reshape(c_xi_yi.shape[0],c_xi_yi.shape[1])
# U_xij_yi= np.array(U_xij_yi).reshape(c_xi_yi.shape[0],c_xi_yi.shape[1])
# emc_xij_yi= np.array(emc_xij_yi).reshape(c_xi_yi.shape[0],c_xi_yi.shape[1])
RO_xij_yi= (U_xij_yi,emc_xij_yi)#395

#helper values
r = np.array(r).reshape(Wb.shape[0],1)
c_rj = enc(r*10**6, pk)

c_rj__xijyi = []
for i in range(len(xi_yi)):#j
    c=[]
    for j in range(len(c_rj)):#type(c_rj[j][0]),type(-xi_yi[i][j])  <class 'int'> <class 'numpy.float64'>
        #c.append(c_rj[j][0]**int(-xi_yi[i][j]*10**1))#直接 指数运算 数据过大溢出报错OverflowError: int too large to convert to float
        c.append(gmpy2.powmod(c_rj[j][0], int(-1 * xi_yi[i][j] * (10 ** 6)), N ** 2) % (N ** 2))
        #decimal?  https://blog.csdn.net/qq_36963214/article/details/108190232
    c_rj__xijyi.append(c)

h__xijyi = []#嵌套列表
for i in range(xi_yi.shape[0]) :#j
    h1 = []
    for j in range (xi_yi.shape[1]):
        h1.append(gp[5]**int(xi_yi[i][j]*10**6))#如果底数是十六进制，指数是10进制，会？       10**2
    h__xijyi.append(h1)

# print('h__xijyi',h__xijyi[0][0]*h__xijyi[0][1]*h__xijyi[0][2]*h__xijyi[0][1]*h__xijyi[0][1],h__xijyi[0][0],h__xijyi[0][0]**int(q_list[0])*h__xijyi[0][0]**int(q_list[0]),q_list[0])
#h__xijyi = np.array(h__xijyi).reshape(xi_yi.shape[0],xi_yi.shape[1])

#DO_TO_SA
DO_TO_SA = ( c_xi_yi, h__xijyi , RO_xij_yi ,c_rj__xijyi )#array list (list,list) list(x,y)
#DO_TO_SB
DO_TO_SB = (h__xijyi , RO_xij_yi)
DO_DS_end = time.perf_counter()

plain_Wb1 = Wb #仅作为flag

#训练相关参数
emcn = 0.001  # a
lr = 0.02  # 学习率
thr = 0.01  # 阈值
T = 100  # 轮次 epoch
t = 0
cost0 = 0
cost = 0
s_sum = 0
SA_time = 0
SB_time = 0

#def train(X_train,y_train,emcn = 0.001,lr = 0.001,thr = 0.001,T = 100):
while(t<T):# np.abs(cost-cost0)>thr or
    cost0 = cost
    cost = 0
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step1'''
    #SA
    SA1_start = time.perf_counter()#!!!
    C = []#for i user  Ci
    for i in range(c_xi_yi.shape[0]):
        Ci1,Ci2=1,1
        for j in range(c_xi_yi.shape[1]):
            #print(type(c_xi_yi[i][j]),type(Wb[j][0]),type(c_rj[j][0]),type(xi_yi[i][j])) #int np.float64 int np.float64
            Ci1 = Ci1 * gmpy2.powmod(c_xi_yi[i][j], int( Wb[j][0] * (10 ** 6)), N ** 2) % (N ** 2)#N**2
            Ci2 = Ci2 * gmpy2.powmod(c_rj[j][0], int(-1 * xi_yi[i][j] * (10 ** 6)), N ** 2) % (N ** 2)
        C.append(Ci1*Ci2)
    C = np.array(C).reshape((1, c_xi_yi.shape[0]))#79

    Partial_C_1 = partialdec(pk, lmd1, C)#部分解密
    SA_to_SB = (C,Partial_C_1)

    SA1_end = time.perf_counter()#!!!
    SA_time = SA_time + (SA1_end - SA1_start)#!!!
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step2'''
    #SB
    Partial_C_2 = partialdec(pk, lmd2, C) #(1,79)
    plain_C = partialcombine(pk, Partial_C_1, Partial_C_2) / 10**12
    plain_C = plain_C.astype(np.float64)
    # #非常规验证 *************************************************************
    real = (xi_yi*10**6).dot(Wb*10**6)/10**12
    #print(Wb,real.shape,real)
    #print('real',real.reshape(real.shape[0],))#针对负数加解密的问题
    Flag = np.array([2*int(x>0)-1 for x in real])
    #print('解密后真实数字（有精度损失）',plain_C * Flag)#加解密时将将负数作为正数处理，最后统一添加符号
    plain_C = plain_C * Flag
    #the verification formula of ci
    verify_Ci=V_correctness_Ci(plain_C,g__wj,h__xijyi)#g__wj 放大了 10**6  h__xijyi 中 int(xi_yi[i][j]*10**6)  plain N**
    #verify_Ci = V_correctness_Ci_V1(plain_C, g__wj, h__xijyi) #不采用批量验证
    #print('Ci验证结果：',verify_Ci)

    #验证成功：
    S = 0
    s = []#空集

    for i in range(len(plain_C[0])):
        if(plain_C[0][i]<1):
            s.append(i)
            S += 1 - plain_C[0][i]#plain_C  shape  (1,79)
    s_sum = s_sum + len(s)
    #print('s,S',s,S)
    #计算密文下的梯度 Cj   针对s
    c_xjy = []
    for j in range(c_xi_yi.shape[1]):#gmpy2.powmod(c_rj[j][0], int(-1 * xi_yi[i][j] * (10 ** 6)), N ** 2) % (N** 2)
        temp = gmpy2.powmod(c_xi_yi[s[0]][j], int(-1 * (emcn * 10 ** 6)), N ** 2)   #按理说本应当指数为负数
        for i in s[1:]:
            temp = temp * gmpy2.powmod( c_xi_yi[i][j], int(-1*(emcn * 10 ** 6)), N ** 2) % (N** 2) #(c_xi_yi[i][j]**(-emcn*10**6))
        c_xjy.append(temp)

    Cj = []
    for j in range(c_Wb.shape[0]):#  c_Wb.shape 5 1
        temp = (gmpy2.powmod(c_Wb[j][0],int(emcn*lr*10**6),N**2)) * (gmpy2.powmod(c_rj[j][0] ,int(-emcn*lr*10**6),N**2)) * c_xjy[j]
        Cj.append(temp)
        #Cj.append(gmpy2.powmod(temp,-1,N**2))
    Cj = np.array(Cj).reshape(1,len(Cj))
    #Cj = Cj**(1*10**(-6)) % N**2

    #密文下更新W
    #c_Wb = enc(Wb*(10**6), pk)
    c_wb1 = []
    for j in range(Cj.shape[1]):#?????????????????????????????
        #c_wb1.append(c_Wb[j][0] *( Cj[j]**(-1))) # 前后密文 放大倍数不一样
        if( plain_Wb1[j]<0):#当被减数为负数时
            #print('case2',pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk))
            c_wb1.append(gmpy2.powmod(c_Wb[j][0], int(-1 * 10 ** 6), N ** 2) * (gmpy2.powmod(Cj[0][j], -1, N ** 2)))
        elif(pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk) < pdec(np.array(gmpy2.powmod(Cj[0][j],-1,N**2)).reshape(1,1),lmd1,lmd2,pk)): #权重由正转为负
            #print('case1',pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk),pdec(np.array(gmpy2.powmod(Cj[0][j],-1,N**2)).reshape(1,1),lmd1,lmd2,pk))#被减数和减数为正，被减数小于减数
            c_wb1.append(gmpy2.powmod(c_Wb[j][0], int(-1 * 10 ** 6), N ** 2) * (gmpy2.powmod(Cj[0][j], -1, N ** 2)))
        else:#3 1,直接相减
            #print('case3')
            c_wb1.append( gmpy2.powmod(c_Wb[j][0],int(1*10**6),N**2) * (gmpy2.powmod(Cj[0][j] ,1,N**2)))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    c_Wb1 = np.array(c_wb1).reshape(c_Wb.shape[0],c_Wb.shape[1])
    #print('c_wb1',c_wb1)
    #计算明文下的权重更新结果进行验证-----------------************************************************************************
    G = (emcn * Wb).reshape(Wb.shape[0],) - emcn * (r.reshape(r.shape[0],))
    #print('初始梯度 emcn * Wb',G)
    for i in s:
        G = G - y_train[i] * X_train[i]
    # print('更新后G',G)
    # print('lr * G',lr * G)#这一步没有问题

    c_xjy = np.array(c_xjy).reshape(len(Wb),1)# 权重长度
    Partial_c_xjy1 = partialdec(pk,lmd1,c_xjy)
    Partial_c_xjy2 = partialdec(pk,lmd2,c_xjy)
    plain_c_xjy = partialcombine(pk,Partial_c_xjy1,Partial_c_xjy2)/(10**12)
    plain_c_xjy = plain_c_xjy.astype(np.float64)
    #print('解密 验证G:plain_c_xjy',plain_c_xjy)#************************************

    plain_Wb1 = Wb.reshape(Wb.shape[0],) - r.reshape(r.shape[0],) - lr * G
    #plain_Wb1 = Wb.reshape(Wb.shape[0], )  - lr * G
    #print('明文下的权重更新结果',plain_Wb1)
    Flag_wb = np.array([2 * int(x > 0) - 1 for x in plain_Wb1])
    # -----------------*******************************************************************************
    #计算cost
    cost=0
#   cost = cost + S + 0.5*emcn*np.linalg.norm(Wb,ord=2)# 求向量模长  默认2范式       暂时用 经扰动后的模型权重
    cost = cost + S + 0.5 * emcn * ((Wb.T).dot(Wb))

    # SB 部分解密
    Partial_Cj_2 = partialdec(pk, lmd2, Cj)#部分解密
    Partial_c_Wb_2 = partialdec(pk, lmd2, c_Wb1)

    SB_TO_SA = [ plain_C, (Cj,Partial_Cj_2), (c_Wb1,Partial_c_Wb_2)] #

    SB1_end = time.perf_counter()  # !!!
    SB_time = SB_time + (SB1_end - SA1_end)
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step3'''
    # SA !!!!!!!!!
    #verify_Ci_2=V_correctness_Ci(SB_TO_SA[0],g__wj,h__xijyi)
    verify_Ci_2 = V_correctness_Ci_V1(SB_TO_SA[0],g__wj,h__xijyi)
    # print('Ci_2验证结果：',verify_Ci_2)
    S2 = 0
    s2 = []#空集
    for i in range(len(plain_C[0])):
        if(plain_C[0][i]<1):
            s2.append(i)
            S2 += 1 - plain_C[0][i]
    #print('s2',s2) 空
    #计算cost
    cost_SA=0
    cost_SA = cost_SA + S2 + 0.5*emcn*np.linalg.norm(Wb,ord=2)# 求向量模长  默认2范式       暂时用 经扰动后的模型权重

    #解密
    Partial_Cj_1 = partialdec(pk, lmd1, Cj)
    Partial_Cj = partialcombine(pk, Partial_Cj_1, Partial_Cj_2) / (10 ** 12)
    Partial_Cj = Partial_Cj.astype(np.float64)
    #print('解密Partial_Cj + r',Partial_Cj,r)  #真实更新

    Partial_c_Wb_1 = partialdec(pk, lmd1, c_Wb1)
    Partial_c_Wb = partialcombine(pk, Partial_c_Wb_1, Partial_c_Wb_2) / (10 ** 12)
    Partial_c_Wb = Partial_c_Wb.astype(np.float64)
    #print('初始权重wb',Wb)
    Wb = Partial_c_Wb.reshape(Partial_c_Wb.shape[0],) * Flag_wb #添加符号
    Wb = Wb.reshape(Partial_c_Wb.shape[0],1)
    #print('明文下的权重更新结果', plain_Wb1)
    #print('解密_更新后Wb1',Wb)  # ***************************************************************************************
    #print('dec',pdec(c_Wb1,lmd1,lmd2,pk))
    print('end : ',t)
    Wb = np.array(plain_Wb1 + r.reshape(r.shape[0],)).reshape(len(plain_Wb1),1)
    #验证 计算tag $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    #(1)cj  MR_TO_SA  DO_TO_SA
    lc_emc_xij_yi = []#U_xij_yi,emc_xij_yi
    for j in range(len(U_xij_yi[0])) :
        #temp = emc_xij_yi[s2[0]][j] % N**2
        temp = U_xij_yi[0][j]**int(-lr*10**6) #% N ** 2#因为S2暂时为空
        for i in range(1,len(U_xij_yi)):
        #for i in s2[1:]:
            temp = temp * U_xij_yi[i][j]**int(-lr*10**6) #% N**2) #  inf 计算溢出
        lc_emc_xij_yi.append(temp)
    #lc_emc_xij_yi = np.array(lc_emc_xij_yi)
    #print('lc_emc_xij_yi',lc_emc_xij_yi,len(lc_emc_xij_yi))
    u_cj = []#np.zeros((len(U_rj),1))
    for j in range(len(U_rj)):
        #print(int(U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])
        u_cj.append((U_wj[j]**int(lr*emcn*10**6)) * (U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])#020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    #print('u_cj',u_cj)

    #h__arai 待修正
    lc_h_arai = []
    for j in range(len(emc_xij_yi[0])) :
        #temp = emc_xij_yi[s2[0]][j] % N**2
        temp = h__arai[0]** int(-lr * emc_xij_yi[0][j] *10**6 ) #因为S2暂时为空  放大10**6
        for i in range(1,len(emc_xij_yi)):
        #for i in s2[1:]:
            temp = temp * (h__arai[i] ** int(-lr * emc_xij_yi[i][j] *10**6 ))#小数  -15411353.18
        lc_h_arai.append(temp)
    emc_cj = []
    for j in range(len(U_rj)):#5
        emc_cj.append( h**int(emcn * lr * emc_wj[j]) * (h**int(-emcn*lr*emc_rj[j]) * lc_h_arai[j])  )#020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    #print('emc_cj',emc_cj)

    #wjt
    u_wjt = []
    for j in range(len(u_cj)):
        u_wjt.append(Element.__ifloordiv__(U_wj[j],u_cj[j])) #除法 用法
    emc_wjt = []
    for j in range(len(U_rj)):#5
        emc_wjt.append(Element.__ifloordiv__(h**int(emc_wj[j]) ,emc_cj[j]))#  202971239619.0  ->  202971239619
    #print('emc_wjt',emc_wjt)

    #cost
    u_cost = (U_W[0] ** int(0.5* emcn * int(K3[0]) ))#U_W,emc_W  放大10**6
    #emc_cost = []
    #emc_cost = ( h**(int(int(MR_a)*(cost+S2)))  *  h**(int(0.5 * emcn * emc_W[0])) )
    print(int(0.5 * emcn * emc_W[0]))
    print(int((MR_a) * (cost + S2)))
    emc_cost = h ** (int(int(MR_a) * (cost + S2))+ (int(0.5 * emcn * emc_W[0]))) #
    #emc_cost = gmpy2.powmod(h, (int(int(MR_a)*(cost+S2))), N ** 2) * gmpy2.powmod(h, int(0.5 * emcn * emc_W[0]) , N ** 2)

    Ro_cj = (u_cj,emc_cj)
    Ro_wjt = (u_wjt,emc_wjt)
    Ro_cost = (u_cost,emc_cost)
    #print('Ro_cost',Ro_cost)
    #验证 the verification formula of cj wjt cost
    #Cj
    verify_Cj = []
    for j in range(len(u_cj)):
        verify_Cj.append(pairing.apply(g,emc_cj[j]) == pairing.apply(g**(int(MR_a)),u_cj[j]*h**int(Cj[0][j])))
    #Wj
    verify_Wj = []
    for j in range(len(u_cj)):
        verify_Wj.append(pairing.apply(g,emc_wjt[j]) == pairing.apply(g**(int(MR_a)),u_wjt[j]*h**int(plain_Wb1[j])))
    #cost
    verify_cost = (pairing.apply(g,emc_cost) == pairing.apply(g**(int(MR_a)),u_cost*h**int(cost)))#取整？

    #验证成功：SA updates
    for j in range(len(Partial_Cj[0])):
        g__wj[j] = g__wj[j] * (g**int(Partial_Cj[0][j]))

    #omega = np.sum(W) #更新
    SA2_end = time.perf_counter()  # !!!
    SA_time = SA_time + (SA2_end - SB1_end)  # !!!
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step4'''
    # #SB
    # SB_TO_MR = Ro_wjt
    # Cj
    verify_Cj_SB = []
    for j in range(len(u_cj)):
        verify_Cj_SB.append(pairing.apply(g, emc_cj[j]) == pairing.apply(g ** (int(MR_a)), u_cj[j] * h ** int(Cj[0][j])))
    #'''5.4 Privacy-preserving and Verifiable Training Protocol:step5'''
    # # #model requester verifies the correctness of wjt
    # # pairing.apply(g, emc_wjt)== pairing.apply()
    #verify_wjt=V_correctness_Ci(SB_TO_SA[0],g__wj,h__xijyi)

    lc_emc_xij_yi = []  # U_xij_yi,emc_xij_yi
    for j in range(len(U_xij_yi[0])):
        # temp = emc_xij_yi[s2[0]][j] % N**2
        temp = U_xij_yi[0][j] ** int(-lr * 10 ** 6)  # % N ** 2#因为S2暂时为空
        for i in range(1, len(U_xij_yi)):
            # for i in s2[1:]:
            temp = temp * U_xij_yi[i][j] ** int(-lr * 10 ** 6)  # % N**2) #  inf 计算溢出
        lc_emc_xij_yi.append(temp)
    # lc_emc_xij_yi = np.array(lc_emc_xij_yi)
    # print('lc_emc_xij_yi',lc_emc_xij_yi,len(lc_emc_xij_yi))
    u_cj = []  # np.zeros((len(U_rj),1))
    for j in range(len(U_rj)):
        # print(int(U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])
        u_cj.append((U_wj[j] ** int(lr * emcn * 10 ** 6)) * (U_rj[j] ** int(-lr * emcn * 10 ** 6)) * lc_emc_xij_yi[
            j])  # 020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    # print('u_cj',u_cj)

    # h__arai 待修正
    lc_h_arai = []
    for j in range(len(emc_xij_yi[0])):
        # temp = emc_xij_yi[s2[0]][j] % N**2
        temp = h__arai[0] ** int(-lr * emc_xij_yi[0][j] * 10 ** 6)  # 因为S2暂时为空  放大10**6
        for i in range(1, len(emc_xij_yi)):
            # for i in s2[1:]:
            temp = temp * (h__arai[i] ** int(-lr * emc_xij_yi[i][j] * 10 ** 6))  # 小数  -15411353.18
        lc_h_arai.append(temp)
    emc_cj = []
    for j in range(len(U_rj)):  # 5
        emc_cj.append(h ** int(emcn * lr * emc_wj[j]) * (h ** int(-emcn * lr * emc_rj[j]) * lc_h_arai[
            j]))  # 020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    # print('emc_cj',emc_cj)

    # wjt
    u_wjt = []
    for j in range(len(u_cj)):
        u_wjt.append(Element.__ifloordiv__(U_wj[j], u_cj[j]))  # 除法 用法
    emc_wjt = []
    for j in range(len(U_rj)):  # 5
        emc_wjt.append(Element.__ifloordiv__(h ** int(emc_wj[j]), emc_cj[j]))  # 202971239619.0  ->  202971239619
    # print('emc_wjt',emc_wjt)

    Ro_cj2 = (u_cj, emc_cj)
    Ro_wjt2 = (u_wjt, emc_wjt)
    # 验证成功：SB updates
    for j in range(len(Partial_Cj[0])):
        g__wj[j] = g__wj[j] * (g ** int(Partial_Cj[0][j]))#该步骤仅用来计算时间

    SB2_end = time.perf_counter()  # !!!
    SB_time = SB_time + (SB2_end - SA2_end)  # !!!

    # # print('最终wjt验证结果：',verify_wjt)
    c_Wb = enc(Wb*(10**6), pk)
    t += 1
'''The end'''
time_end = time.perf_counter()
print("time cost %f:" % (time_end-time_start))

#预测效果验证：
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
w = plain_Wb1#明文下的计算结果
w1 = plain_Wb1.reshape(r.shape[0],) - r.reshape(r.shape[0],)
#print('plain_Wb1 - r',plain_Wb1,r.reshape(r.shape[0],))
#w = Wb#负数更新存在问题
#predict_y = list((X_test).dot(w.T))
predict_y = list((X_test).dot(w.reshape(r.shape[0],).T))
predict_y1 = list((X_test).dot(w1.T))
#print('predict_y1',predict_y1)
for x in  range(len(predict_y)):
    if(predict_y[x] >= 1):
        predict_y[x] = 1
    else:
        predict_y[x]=0
test_predictions = predict_y#预测 标签

for x in  range(len(predict_y1)):
    if(predict_y1[x] >= 1):
        predict_y1[x] = 1
    else:
        predict_y1[x]=0
test_predictions1 = predict_y1#预测 标签

y_true = list(y_test)
#print('真实值',y_true)
#print('预测值',test_predictions)
acc = accuracy_score(y_true,test_predictions)
recall = recall_score(y_true,test_predictions,average='macro')
precision = precision_score(y_true,test_predictions,average='macro')
f1 = f1_score(y_true,test_predictions,average='macro')
#print('macro -> accuracy: %.4f recall: %.4f precision:%.4f f1: %.4f'%(acc,recall,precision,f1))

#扰动值
acc = accuracy_score(y_true,test_predictions1)
recall = recall_score(y_true,test_predictions1,average='macro')
precision = precision_score(y_true,test_predictions1,average='macro')
f1 = f1_score(y_true,test_predictions1,average='macro')
print('扰动后macro -> accuracy: %.4f recall: %.4f precision:%.4f f1: %.4f'%(acc,recall,precision,f1))
#print('s_sum',s_sum)

print("MR 数据上传时间 time cost %f:" % (- MR_DS_start + MR_DS_end))
print("DO 数据上传时间 time cost %f:" % (- DO_DS_start + DO_DS_end))

print("Cloud A 时间 time cost %f:" % SA_time)
print("Cloud B 时间 time cost %f:" % SB_time)
