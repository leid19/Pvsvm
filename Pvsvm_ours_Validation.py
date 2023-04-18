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
    # data load
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_image01 = train_images[np.where(train_labels <= 1)[0]]
    train_label01 = train_labels[np.where(train_labels <= 1)[0]]

    train_image01_flat = train_image01.reshape(train_image01.shape[0], 784)
    print('Sample number', train_image01.shape[0])
    # PCA dimension reduction
    # Create a model
    pca = PCA(n_components=20)  # n_components
    # #train_image01_flat.shape
    reduced_train_image01_flat = pca.fit_transform(train_image01_flat)
    print(reduced_train_image01_flat[0])
    one = np.zeros(len(reduced_train_image01_flat)) + 1
    reduced_train_image01_flat = np.column_stack((reduced_train_image01_flat, one))  # Extended dimension
    print(reduced_train_image01_flat[0])
    # Divide the train set and test set
    X_train, X_test, y_train, y_test = train_test_split(reduced_train_image01_flat, train_label01, test_size=0.3,
                                                        shuffle=True, random_state=3)
    # 
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    X_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return  X_train, X_test, y_train, y_test

def get_iris_data():# Load the iris dataset
    iris = datasets.load_iris()
    #The size of iris.data is 150*4, representing 4 features. Here, we can only choose the last two types of features x = iris.data[:, [2, 3]]
    x = iris.data
    y = iris.target
    first_id = list(y).index(2)
    #x = np.trunc(x[:first_id - 1])# round
    x = x[:first_id - 1]  
    one = np.zeros(len(x))+1
    x = np.column_stack((x,one))#Extended dimension
    #print(x, type(x))
    y = y[:first_id - 1]#observe data and choose label：0，1
    y = -(y*(-2)+1)#translate label:0 to label:-1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=2)
    x_train = x_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = x_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return  x_train, x_test, y_train, y_test

def get_heart_data():
    # data loader
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

    # Convert discrete categorical and ordinal feature columns into One-Hot encoding, and expand categorical data into features
    df = pd.get_dummies(df)
    # df.head(3)

    y = df.target.values
    y = (y * (2) - 1)  

    X = df.drop(['target'], axis=1)
    X = np.array(X)
    # pca = PCA()
    # pca.fit(X)
    # X = pca.transform(X)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)  # 13
    # Data standardization
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one))  # Extended dimension
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one))  # Extended dimension
    x_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.int64)
    x_test = X_test.astype(np.float64)
    y_test = y_test.astype(np.int64)
    return x_train, x_test, y_train, y_test

def get_breast_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 1

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one))  
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one)) 
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

        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            # Set the appropriate row in our dataset
            X[i] = data
            # 将“g”记为1，将“b”记为0。
            y[i] = bool(row[-1] == 'g')
    y = 2 * y.astype(int) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    one = np.zeros(len(X_train)) + 1
    X_train = np.column_stack((X_train, one)) 
    one = np.zeros(len(X_test)) + 1
    X_test = np.column_stack((X_test, one)) 
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
    s = (lmd * gmpy2.invert(lmd,n**2)) % (lmd * (n ** 2))#Modular inverse operation

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

def V_correctness_Ci(plain_C,g__wj,h__xijyi):#batch verification
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

def V_correctness_Ci_V1(plain_C,g__wj,h__xijyi):#normal verification
    ver_list = []
    for i in range(plain_C.shape[1]):
        e1 = pairing.apply(g ** int(plain_C[0][i]), h)#heart (0,237)
        temp_first = pairing.apply(g__wj[0], h__xijyi[i][0])
        for j in range(len(h__xijyi[0])):  # 5
            temp_first = temp_first * pairing.apply(g__wj[j], h__xijyi[i][j])

        ver_list.append( temp_first == e1)
    return all(ver_list)

# print('Bilinear pairing-based accumulative verification',pairing.apply(g**2,h**2),pairing.apply(g,h)*pairing.apply(g,h))

#***********************************************************************************************************************
time_start = time.perf_counter()

#Get initial data
#X_train, X_test, y_train, y_test = get_iris_data() # Iris
#X_train, X_test, y_train, y_test = get_heart_data() # Heart disease
#X_train, X_test, y_train, y_test = get_breast_data() # Breast cancer
#X_train, X_test, y_train, y_test = get_Ionosphere_data() # Ionosphere
X_train, X_test, y_train, y_test = get_minist_data()#minist

'''Initialization_Step1_TTP'''
# #Key_Gen
pk, sk = keygen()#sk randomness
lmd1, lmd2 = secrekeySplit(pk,sk)
PK = pk[1]
N = pk[0]

#***********************************************************************************************************************
#Construct bilinear mapping pypbc
# Instantiate based on security parameters. https://www.freesion.com/article/19881431966/
# qbits=512
# rbits=160
# params = Parameters(qbits=qbits, rbits=rbits)   #Parameter initialization should choose type-A curves to ensure that G1 X G1 = G2.


#Generate curves based on the set order of the group.
q_1 = get_random_prime(20) # Generate a random prime number of length 20
q_2 = get_random_prime(20)
p = q_1 * q_2 # k
#Use the a1_param parameter in pbc.
params = Parameters(n=p)
pairing = Pairing(params) # Instantiate bilinear pairing based on the parameters.
#Take a random number from the group and initialize an element, usually g, which is a generator of G1.
g = Element.random(pairing, G1) # g and h are generators of G1
h = Element.random(pairing, G1) # type(G1): <class 'int'>


#Bilinear pairing operation as follows: #e(g, g)
#e = pairing.apply(g, g)
#Generally, group operations are performed using Element rather than direct numerical operations. (Of course, direct operations are also possible.)
#Here, "pairing" represents the bilinear pairing initialized, G2 represents the return value type, "value" represents the value of the element, G2 elements act as the base, and Zr elements act as the exponent.
#Actually, using b = g ** c gives the same effect, but writing it in the following way is neater and clearer, reducing errors.
#b = Element(pairing, G2, value=g ** c) # b = g^c

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
#Randomly initialize weights
#print(np.random.normal(loc=0, scale=0.01, size=(5,2)).astype(np.float64))
#W = np.random.randn(len(X_train[0]))#  [round(x) for x in np.random.normal(0,2,4)]
np.random.seed(1)
W = np.random.normal( loc=0, scale=0.01, size=(len(X_train[0]),) ).astype(np.float64)
#W = np.array([1,1,1,1,1]).astype(np.float64)# Given a fixed value
W[-1]=np.float64(1.0)
omega = np.sum(W)
#print('omega',omega)
#W=W*(10**8))#Amplify and round up

#Randomly initialize perturbation values
r = np.random.normal(loc=0, scale=0.01, size=(len(X_train[0]),)).astype(np.float64)
r[-1]=np.float64(0)
#r = r*(10**8)
#r=[round(i) for i in r]#

#add perturbation values
Wb =  np.sum([W, r], axis=0)

K_list = np.random.randint(0,p,size=(2,len(X_train[0])))#range of Zp？****************************    h?
K1,K2= K_list[0].astype(int),K_list[1].astype(int)
K3 = np.random.randint(0,p,size=(1))

#print('Verify the bilinear mapping.', pairing.apply(g ** K1, h ** K2)== pairing.apply(g, h) ** (K1 * K2))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
U_wj = [ (gp[-1]**int(x)) for x in K1]#h  list
U_rj = [gp[-1]**int(x) for x in K2]
U_W = [gp[-1]**int(K3[0])]# omega list

#tag
# print(MR_a,int(MR_a),Wb[0],K1,p)
# print(type(MR_a),type(Wb[0]),type(K1),type(p))
emc_wj = [(int(MR_a)*(Wb[j]+K1[j]))%p for j in range(len(Wb))]   #print(1**0x11095A059DA86A==1**8908973441865092) true(As hexadecimal and decimal representations are equivalent when used as exponents.)
#omg = [w**2 for w in W]
#emc_W = [(int(MR_a)*(omg[j]+K2[j]))%p for j in range(len(omg))]#
emc_rj = [(int(MR_a)*(r[j]+K2[j]))%p for j in range(len(r))]#
emc_W  = [(int(MR_a)*(omega+K3[0]))%p ]

# the model requester : helper values
g__wj = [gp[4]**int(wj*10**6) for wj in W]#   W*10**6   actually should be 10**8
h__arai = [(gp[5]**(MR_a))**(DO_ai**(-1)) for DO_ai in DO_a]#Pay attention to the data type, whether it will cause precision loss
# the model requester : encrypts the model weights
Wb = np.array(Wb).reshape(Wb.shape[0],1)#8->int
c_Wb = enc(Wb*(10**6), pk)#For ease of calculation and writing, the list can be converted to an array type.  shape(5,1)

#send:MR_SA
MR_TO_SA=[ (g__wj,Wb,(U_wj,emc_wj)) , (U_rj,emc_rj) , h__arai , (U_W,emc_W) ]
#Not arranged in the exact order as in the article.
#send:MR_SB
MR_TO_SB=[( g__wj , r , (U_rj,emc_rj) , (U_wj,emc_wj) , c_Wb ),h__arai]
#***********************************************************************************************************************
MR_DS_end = time.perf_counter()

'''Data Submission_Step2:Data owner'''
DO_DS_start = time.perf_counter()
#extends his/her training data，This operation has been performed during data initialization (before data partitioning).

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
        u.append(gp[5] ** kij)  # **************************data type:element int
        emc.append((int(DO_a[x]) * (xi_yi[x][y] + kij)) % p)#Convert DO_a[x] (selected from Zp) from hexadecimal to decimal first
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
        #c.append(c_rj[j][0]**int(-xi_yi[i][j]*10**1))#Direct exponentiation causes overflow error when the data is too large.  :  OverflowError: int too large to convert to float
        c.append(gmpy2.powmod(c_rj[j][0], int(-1 * xi_yi[i][j] * (10 ** 6)), N ** 2) % (N ** 2))
        #decimal?  https://blog.csdn.net/qq_36963214/article/details/108190232
    c_rj__xijyi.append(c)

h__xijyi = []#Nested List
for i in range(xi_yi.shape[0]) :#j
    h1 = []
    for j in range (xi_yi.shape[1]):
        h1.append(gp[5]**int(xi_yi[i][j]*10**6))
    h__xijyi.append(h1)

# print('h__xijyi',h__xijyi[0][0]*h__xijyi[0][1]*h__xijyi[0][2]*h__xijyi[0][1]*h__xijyi[0][1],h__xijyi[0][0],h__xijyi[0][0]**int(q_list[0])*h__xijyi[0][0]**int(q_list[0]),q_list[0])
#h__xijyi = np.array(h__xijyi).reshape(xi_yi.shape[0],xi_yi.shape[1])

#DO_TO_SA
DO_TO_SA = ( c_xi_yi, h__xijyi , RO_xij_yi ,c_rj__xijyi )#array list (list,list) list(x,y)
#DO_TO_SB
DO_TO_SB = (h__xijyi , RO_xij_yi)
DO_DS_end = time.perf_counter()

plain_Wb1 = Wb #as a flag

#Training related parameter
emcn = 0.001  # a
lr = 0.02  # learning rate
thr = 0.01  # threshold
T = 100  # epoch
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

    Partial_C_1 = partialdec(pk, lmd1, C)#Partial decryption
    SA_to_SB = (C,Partial_C_1)

    SA1_end = time.perf_counter()#!!!
    SA_time = SA_time + (SA1_end - SA1_start)#!!!
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step2'''
    #SB
    Partial_C_2 = partialdec(pk, lmd2, C) #(1,79)
    plain_C = partialcombine(pk, Partial_C_1, Partial_C_2) / 10**12
    plain_C = plain_C.astype(np.float64)
    # #verification *************************************************************
    real = (xi_yi*10**6).dot(Wb*10**6)/10**12
    Flag = np.array([2*int(x>0)-1 for x in real])
    plain_C = plain_C * Flag
    # the verification formula of ci
    verify_Ci=V_correctness_Ci(plain_C,g__wj,h__xijyi)#g__wj is amplified by 106, h__xijyi contains int(xi_yi[i][j]*106) plain N**
    #verify_Ci = V_correctness_Ci_V1(plain_C, g__wj, h__xijyi) # do not use batch verification
    #print('Ci verification result:',verify_Ci)
    
    #success：
    S = 0
    s = []#Empty set

    for i in range(len(plain_C[0])):
        if(plain_C[0][i]<1):
            s.append(i)
            S += 1 - plain_C[0][i]#plain_C  shape  (1,79)
    s_sum = s_sum + len(s)
    #print('s,S',s,S)
    #Calculate the gradient Cj under ciphertext for s.
    c_xjy = []
    for j in range(c_xi_yi.shape[1]):#gmpy2.powmod(c_rj[j][0], int(-1 * xi_yi[i][j] * (10 ** 6)), N ** 2) % (N** 2)
        temp = gmpy2.powmod(c_xi_yi[s[0]][j], int(-1 * (emcn * 10 ** 6)), N ** 2)   
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

    #Update W in ciphertext
    #c_Wb = enc(Wb*(10**6), pk)
    c_wb1 = []
    for j in range(Cj.shape[1]):#?????????????????????????????
        #c_wb1.append(c_Wb[j][0] *( Cj[j]**(-1))) 
        if( plain_Wb1[j]<0):#When the minus number is negative
            #print('case2',pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk))
            c_wb1.append(gmpy2.powmod(c_Wb[j][0], int(-1 * 10 ** 6), N ** 2) * (gmpy2.powmod(Cj[0][j], -1, N ** 2)))
        elif(pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk) < pdec(np.array(gmpy2.powmod(Cj[0][j],-1,N**2)).reshape(1,1),lmd1,lmd2,pk)): #权重由正转为负
            #print('case1',pdec(np.array(gmpy2.powmod(c_Wb[j][0],1*10**6,N**2)).reshape(1,1),lmd1,lmd2,pk),pdec(np.array(gmpy2.powmod(Cj[0][j],-1,N**2)).reshape(1,1),lmd1,lmd2,pk))#被减数和减数为正，被减数小于减数
            c_wb1.append(gmpy2.powmod(c_Wb[j][0], int(-1 * 10 ** 6), N ** 2) * (gmpy2.powmod(Cj[0][j], -1, N ** 2)))
        else:#3 1,Direct subtraction
            #print('case3')
            c_wb1.append( gmpy2.powmod(c_Wb[j][0],int(1*10**6),N**2) * (gmpy2.powmod(Cj[0][j] ,1,N**2)))#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    c_Wb1 = np.array(c_wb1).reshape(c_Wb.shape[0],c_Wb.shape[1])
    #print('c_wb1',c_wb1)
    #Validate the weight update results under plaintext.-----------------************************************************************************
    G = (emcn * Wb).reshape(Wb.shape[0],) - emcn * (r.reshape(r.shape[0],))
    for i in s:
        G = G - y_train[i] * X_train[i]

    c_xjy = np.array(c_xjy).reshape(len(Wb),1)#  length of weight
    Partial_c_xjy1 = partialdec(pk,lmd1,c_xjy)
    Partial_c_xjy2 = partialdec(pk,lmd2,c_xjy)
    plain_c_xjy = partialcombine(pk,Partial_c_xjy1,Partial_c_xjy2)/(10**12)
    plain_c_xjy = plain_c_xjy.astype(np.float64)
    
    plain_Wb1 = Wb.reshape(Wb.shape[0],) - r.reshape(r.shape[0],) - lr * G

    Flag_wb = np.array([2 * int(x > 0) - 1 for x in plain_Wb1])
    # -----------------*******************************************************************************
    #calculate cost
    cost=0
#   cost = cost + S + 0.5*emcn*np.linalg.norm(Wb,ord=2)#  Calculate the magnitude of the vector, default 2-norm.
    cost = cost + S + 0.5 * emcn * ((Wb.T).dot(Wb))

    # SB Partial decryption
    Partial_Cj_2 = partialdec(pk, lmd2, Cj)#Partial decryption
    Partial_c_Wb_2 = partialdec(pk, lmd2, c_Wb1)

    SB_TO_SA = [ plain_C, (Cj,Partial_Cj_2), (c_Wb1,Partial_c_Wb_2)] #

    SB1_end = time.perf_counter()  # !!!
    SB_time = SB_time + (SB1_end - SA1_end)
    '''5.4 Privacy-preserving and Verifiable Training Protocol:step3'''
    # SA !!!!!!!!!
    #verify_Ci_2=V_correctness_Ci(SB_TO_SA[0],g__wj,h__xijyi)
    verify_Ci_2 = V_correctness_Ci_V1(SB_TO_SA[0],g__wj,h__xijyi)
    # print('verification result of Ci_2：',verify_Ci_2)
    S2 = 0
    s2 = []#empty set
    for i in range(len(plain_C[0])):
        if(plain_C[0][i]<1):
            s2.append(i)
            S2 += 1 - plain_C[0][i]
    #print('s2',s2) 空
    #cost
    cost_SA=0
    cost_SA = cost_SA + S2 + 0.5*emcn*np.linalg.norm(Wb,ord=2)#Calculate the magnitude of the vector, default 2-norm.

    #Decryption
    Partial_Cj_1 = partialdec(pk, lmd1, Cj)
    Partial_Cj = partialcombine(pk, Partial_Cj_1, Partial_Cj_2) / (10 ** 12)
    Partial_Cj = Partial_Cj.astype(np.float64)
    

    Partial_c_Wb_1 = partialdec(pk, lmd1, c_Wb1)
    Partial_c_Wb = partialcombine(pk, Partial_c_Wb_1, Partial_c_Wb_2) / (10 ** 12)
    Partial_c_Wb = Partial_c_Wb.astype(np.float64)

    Wb = Partial_c_Wb.reshape(Partial_c_Wb.shape[0],) * Flag_wb #Add symbol
    Wb = Wb.reshape(Partial_c_Wb.shape[0],1)

    print('end : ',t)
    Wb = np.array(plain_Wb1 + r.reshape(r.shape[0],)).reshape(len(plain_Wb1),1)
    #verification: calculate tag $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    #(1)cj  MR_TO_SA  DO_TO_SA
    lc_emc_xij_yi = []#U_xij_yi,emc_xij_yi
    for j in range(len(U_xij_yi[0])) :
        #temp = emc_xij_yi[s2[0]][j] % N**2
        temp = U_xij_yi[0][j]**int(-lr*10**6) #% N ** 2
        for i in range(1,len(U_xij_yi)):
        #for i in s2[1:]:
            temp = temp * U_xij_yi[i][j]**int(-lr*10**6) #% N**2) #  inf Computation overflow
        lc_emc_xij_yi.append(temp)
    #lc_emc_xij_yi = np.array(lc_emc_xij_yi)
    #print('lc_emc_xij_yi',lc_emc_xij_yi,len(lc_emc_xij_yi))
    u_cj = []#np.zeros((len(U_rj),1))
    for j in range(len(U_rj)):
        #print(int(U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])
        u_cj.append((U_wj[j]**int(lr*emcn*10**6)) * (U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])#020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    #print('u_cj',u_cj)

    #h__arai  :To be corrected
    lc_h_arai = []
    for j in range(len(emc_xij_yi[0])) :
        #temp = emc_xij_yi[s2[0]][j] % N**2
        temp = h__arai[0]** int(-lr * emc_xij_yi[0][j] *10**6 ) #10**6
        for i in range(1,len(emc_xij_yi)):
        #for i in s2[1:]:
            temp = temp * (h__arai[i] ** int(-lr * emc_xij_yi[i][j] *10**6 ))#decimal  -15411353.18
        lc_h_arai.append(temp)
    emc_cj = []
    for j in range(len(U_rj)):#5
        emc_cj.append( h**int(emcn * lr * emc_wj[j]) * (h**int(-emcn*lr*emc_rj[j]) * lc_h_arai[j])  )#020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    #print('emc_cj',emc_cj)

    #wjt
    u_wjt = []
    for j in range(len(u_cj)):
        u_wjt.append(Element.__ifloordiv__(U_wj[j],u_cj[j])) #Division
    emc_wjt = []
    for j in range(len(U_rj)):#5
        emc_wjt.append(Element.__ifloordiv__(h**int(emc_wj[j]) ,emc_cj[j]))#  202971239619.0  ->  202971239619
    #print('emc_wjt',emc_wjt)

    #cost
    u_cost = (U_W[0] ** int(0.5* emcn * int(K3[0]) ))#U_W,emc_W  Magnification：10**6
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
    # the verification formula of cj wjt cost
    #Cj
    verify_Cj = []
    for j in range(len(u_cj)):
        verify_Cj.append(pairing.apply(g,emc_cj[j]) == pairing.apply(g**(int(MR_a)),u_cj[j]*h**int(Cj[0][j])))
    #Wj
    verify_Wj = []
    for j in range(len(u_cj)):
        verify_Wj.append(pairing.apply(g,emc_wjt[j]) == pairing.apply(g**(int(MR_a)),u_wjt[j]*h**int(plain_Wb1[j])))
    #cost
    verify_cost = (pairing.apply(g,emc_cost) == pairing.apply(g**(int(MR_a)),u_cost*h**int(cost)))

    #success：SA updates
    for j in range(len(Partial_Cj[0])):
        g__wj[j] = g__wj[j] * (g**int(Partial_Cj[0][j]))

    #omega = np.sum(W) #update
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
        temp = U_xij_yi[0][j] ** int(-lr * 10 ** 6)  # % N ** 2#
        for i in range(1, len(U_xij_yi)):
            # for i in s2[1:]:
            temp = temp * U_xij_yi[i][j] ** int(-lr * 10 ** 6)  # % N**2) #  inf overflow
        lc_emc_xij_yi.append(temp)
    # lc_emc_xij_yi = np.array(lc_emc_xij_yi)
    # print('lc_emc_xij_yi',lc_emc_xij_yi,len(lc_emc_xij_yi))
    u_cj = []  # np.zeros((len(U_rj),1))
    for j in range(len(U_rj)):
        # print(int(U_rj[j] **int(-lr*emcn*10**6)) * lc_emc_xij_yi[j])
        u_cj.append((U_wj[j] ** int(lr * emcn * 10 ** 6)) * (U_rj[j] ** int(-lr * emcn * 10 ** 6)) * lc_emc_xij_yi[
            j])  # 020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    # print('u_cj',u_cj)

    # h__arai 
    lc_h_arai = []
    for j in range(len(emc_xij_yi[0])):
        # temp = emc_xij_yi[s2[0]][j] % N**2
        temp = h__arai[0] ** int(-lr * emc_xij_yi[0][j] * 10 ** 6)  
        for i in range(1, len(emc_xij_yi)):
            # for i in s2[1:]:
            temp = temp * (h__arai[i] ** int(-lr * emc_xij_yi[i][j] * 10 ** 6)) 
        lc_h_arai.append(temp)
    emc_cj = []
    for j in range(len(U_rj)):  # 5
        emc_cj.append(h ** int(emcn * lr * emc_wj[j]) * (h ** int(-emcn * lr * emc_rj[j]) * lc_h_arai[
            j]))  # 020019FA460B84 1 020019FA460B84 1.0765643415944357e+906
    # print('emc_cj',emc_cj)

    # wjt
    u_wjt = []
    for j in range(len(u_cj)):
        u_wjt.append(Element.__ifloordiv__(U_wj[j], u_cj[j]))  # division
    emc_wjt = []
    for j in range(len(U_rj)):  # 5
        emc_wjt.append(Element.__ifloordiv__(h ** int(emc_wj[j]), emc_cj[j]))  # 202971239619.0  ->  202971239619
    # print('emc_wjt',emc_wjt)

    Ro_cj2 = (u_cj, emc_cj)
    Ro_wjt2 = (u_wjt, emc_wjt)
    # success：SB updates
    for j in range(len(Partial_Cj[0])):
        g__wj[j] = g__wj[j] * (g ** int(Partial_Cj[0][j]))#time

    SB2_end = time.perf_counter()  # !!!
    SB_time = SB_time + (SB2_end - SA2_end)  # !!!

   
    c_Wb = enc(Wb*(10**6), pk)
    t += 1
'''The end'''
time_end = time.perf_counter()
print("time cost %f:" % (time_end-time_start))

#Prediction effect verification：
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
w = plain_Wb1
w1 = plain_Wb1.reshape(r.shape[0],) - r.reshape(r.shape[0],)

predict_y = list((X_test).dot(w.reshape(r.shape[0],).T))
predict_y1 = list((X_test).dot(w1.T))
#print('predict_y1',predict_y1)
for x in  range(len(predict_y)):
    if(predict_y[x] >= 1):
        predict_y[x] = 1
    else:
        predict_y[x]=0
test_predictions = predict_y#prediction: label

for x in  range(len(predict_y1)):
    if(predict_y1[x] >= 1):
        predict_y1[x] = 1
    else:
        predict_y1[x]=0
test_predictions1 = predict_y1#prediction: label

y_true = list(y_test)
#print('true value',y_true)
#print('prediction',test_predictions)
acc = accuracy_score(y_true,test_predictions)
recall = recall_score(y_true,test_predictions,average='macro')
precision = precision_score(y_true,test_predictions,average='macro')
f1 = f1_score(y_true,test_predictions,average='macro')
#print('macro -> accuracy: %.4f recall: %.4f precision:%.4f f1: %.4f'%(acc,recall,precision,f1))

#Disturbance value
acc = accuracy_score(y_true,test_predictions1)
recall = recall_score(y_true,test_predictions1,average='macro')
precision = precision_score(y_true,test_predictions1,average='macro')
f1 = f1_score(y_true,test_predictions1,average='macro')
print('Disturbance：macro -> accuracy: %.4f recall: %.4f precision:%.4f f1: %.4f'%(acc,recall,precision,f1))
#print('s_sum',s_sum)

print("MR data submission time cost %f:" % (- MR_DS_start + MR_DS_end))
print("DO data submission time cost %f:" % (- DO_DS_start + DO_DS_end))

print("Cloud A  time cost %f:" % SA_time)
print("Cloud B  time cost %f:" % SB_time)
