from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def data_train():
    num_input=70*45
    labels1=np.loadtxt("labels/labels_new4.csv", delimiter=",")
    labels1=labels1[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_training/data_new4/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_training/data_new4/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data1=np.zeros(shape=[3150,50,50,2])
    data1[:,:,:,0]=img1
    data1[:,:,:,1]=img2
    data1=(data1/250-1)*(-1)
    
    ###############################
    labels2=np.loadtxt("labels/labels_new6.csv", delimiter=",")
    labels2=labels2[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_training/data_new6/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_training/data_new6/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data2=np.zeros(shape=[3150,50,50,2])
    data2[:,:,:,0]=img1
    data2[:,:,:,1]=img2
    data2=(data2/250-1)*(-1)
    #################################
    labels3=np.loadtxt("labels/labels_new7.csv", delimiter=",")
    labels3=labels3[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_training/data_new7/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_training/data_new7/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data3=np.zeros(shape=[3150,50,50,2])
    data3[:,:,:,0]=img1
    data3[:,:,:,1]=img2
    data3=(data3/250-1)*(-1)
    ###################################
    labels4=np.loadtxt("labels/labels_new8.csv", delimiter=",")
    labels4=labels4[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_training/data_new8/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_training/data_new8/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data4=np.zeros(shape=[3150,50,50,2])
    data4[:,:,:,0]=img1
    data4[:,:,:,1]=img2
    data4=(data4/250-1)*(-1)
    
    ###################################
    labels5=np.loadtxt("labels/labels_new9.csv", delimiter=",")
    labels5=labels4[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_training/data_new9/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_training/data_new9/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data5=np.zeros(shape=[3150,50,50,2])
    data5[:,:,:,0]=img1
    data5[:,:,:,1]=img2
    data5=(data5/250-1)*(-1)
    
    #####################################
    # Define Training data 
    x_data=(np.array(np.concatenate((data1,data2,data3,data4,data5)))**2)*250-127
    x_data = x_data.astype('float32')
    x_data /= 255
    y_data=np.array(np.concatenate((labels1,labels2,labels3,labels4,labels5)))

    return (x_data,y_data)


def testing():
    num_input=70*45
    labels1=np.loadtxt("labels/labels_new10.csv", delimiter=",")
    labels1=labels1[:num_input]
    img1=[]
    img2=[]
    for i in range(70):

        for j in range(45):
            image=Image.open("data_testing/data_new10/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_testing/data_new10/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data1=np.zeros(shape=[3150,50,50,2])
    data1[:,:,:,0]=img1
    data1[:,:,:,1]=img2
    data1=(data1/250-1)*(-1)
    ###############################
    # Define Testing data 
    x_data2=(np.array((data1))**2)*250-127
    x_data2 = x_data2.astype('float32')
    x_data2 /= 255
    y_data2=np.array((labels1))
    
    
    return (x_data2,y_data2)

def marmousi():
    num_input=385*45
    labels0=np.loadtxt("labels/labels_marmousi.csv", delimiter=",")
    img1=[]
    img2=[]
    for i in range(385):

        for j in range(45):
            image=Image.open("data_marmousi/data_1.%03d.png"%(i)).convert('L')
            img1.append(np.reshape(image,(50,50)))
            image2=Image.open("data_marmousi/data_2.%03d-%03d.png"%(i,j)).convert('L')
            img2.append(np.reshape(image2,(50,50)))
            
    data0=np.zeros(shape=[17325,50,50,2])
    data0[:,:,:,0]=img1
    data0[:,:,:,1]=img2
    data0=(data0/250-1)*(-1)
    x_data_marm=((np.array(data0)**2)*250)-127
    x_data_marm = x_data_marm.astype('float32')
    x_data_marm /= 255
    y_data_marm=np.array(labels0)
    
    TLdata_size=38
    x_data_marm_tl=np.zeros(shape=(TLdata_size*45,50,50,2))
    y_data_marm_tl=np.zeros(shape=(TLdata_size*45,40))
    for i in range(TLdata_size):
        a=int(385/TLdata_size)*i
        x_data_marm_tl[i*45:(i+1)*45,:,:,:]=x_data_marm[a*45:(a+1)*45,:,:,:]
        y_data_marm_tl[i*45:(i+1)*45,:]=y_data_marm[a*45:(a+1)*45,:]
    
    return (x_data_marm,y_data_marm,x_data_marm_tl,y_data_marm_tl)
