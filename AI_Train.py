from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adagrad
from scipy.io import savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import os
from os import listdir
import numpy as np
import numpy
from datetime import datetime

def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

def load_model(json_path): 
    model = model_from_json(open(json_path).read())
    return model

def load_dataset_Train_batch(AbnormalPath, NormalPath):
  
    batchsize=60
    n_exp= int(batchsize/2)

    Num_abnormal = 900
    Num_Normal = 792


    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal-n_exp:]
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal-n_exp:]
    
    All_Videos=[]
    with open(AbnormalPath+"anomaly.txt", 'r') as f1: #file contain path to anomaly video file.
      for line in f1:
          All_Videos.append(line.strip())
    AllFeatures = []
    print("Loading Anomaly videos Features...")

    Video_count=-1
    for iv in Abnor_list_iter:
        Video_count=Video_count+1
        VideoPath = os.path.join(AbnormalPath, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) / 4096
        
        count = -1;
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))

        if Video_count == 0:
            AllFeatures = VideoFeatues
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatues))
    print("Abnormal Features loaded")

    All_Videos=[]
    with open(NormalPath+"normal.txt", 'r') as f1: #file contain path to normal video file.
        for line in f1:
            All_Videos.append(line.strip())
    
    print("Loading Normal videos...")
  
    for iv in Norm_list_iter:
        VideoPath = os.path.join(NormalPath, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(words) /4096
        count = -1;
        VideoFeatues = []
        for feat in range(0, int(num_feat)):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
            feat_row1 = []
        AllFeatures = np.vstack((AllFeatures, VideoFeatues))

    print("Features loaded")

    AllLabels = np.zeros(32*batchsize, dtype='uint8')
    th_loop1=n_exp*32
    th_loop2=n_exp*32-1

    for iv in range(0, 32*batchsize):
            if iv< th_loop1:
                AllLabels[iv] = int(0)
            if iv > th_loop2:
                AllLabels[iv] = int(1)

    return  AllFeatures,AllLabels


def custom_objective(y_true, y_pred):

    y_true = T.flatten(y_true)
    y_pred = T.flatten(y_pred)
   
    n_seg = 32
    nvid = 60
    n_exp = nvid / 2
    Num_d=32*nvid

    sub_max = T.ones_like(y_pred)
    sub_sum_labels = T.ones_like(y_true)
    sub_sum_l1=T.ones_like(y_true) 
    sub_l2 = T.ones_like(y_true)

    for ii in range(0, nvid, 1):
      
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])

        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])  
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    sub_score = sub_max[Num_d:]
    F_labels = sub_sum_labels[Num_d:]
    

    sub_sum_l1 = sub_sum_l1[Num_d:]
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]
    sub_l2 = sub_l2[:n_exp]

    indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor=n_exp

    Sub_Nor = sub_score[indx_nor]
    Sub_Abn = sub_score[indx_abn]

    z = T.ones_like(y_true)
    for ii in range(0, n_Nor, 1):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]
    z = T.mean(z, axis=-1) +  0.00008*T.sum(sub_sum_l1) + 0.00008*T.sum(sub_l2)

    return z




# Path contains C3D features (.txt file) of each video.
# Each text file contains 32 features, each of 4096 dimension
AllClassPath='C:\\Users\\Var\\Downloads\\Compressed\\anomaly-detection-master\\anomaly-detection-master\\out\\'

output_dir='C:\\Users\\Var\\Downloads\\Compressed\\anomaly-detection-master\\anomaly-detection-master\\trained_model\\'

# Output_dir save trained weights and model.

weights_path = output_dir + 'weights.mat'

model_path = output_dir + 'model.json'



#Create Full connected Model
model = Sequential()
model.add(Dense(512, input_dim=4096,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1,kernel_initializer='glorot_normal',kernel_regularizer=l2(0.001),activation='sigmoid'))

adagrad=Adagrad(lr=0.01, epsilon=1e-08)

model.compile(loss=custom_objective, optimizer=adagrad)

if not os.path.exists(output_dir):
       os.makedirs(output_dir)

All_class_files= listdir(AllClassPath)
All_class_files.sort()
loss_graph =[]
num_iters = 6
total_iterations = 0
batchsize=60
time_before = datetime.now()

for it_num in range(num_iters):
    inputs, targets=load_dataset_Train_batch(AllClassPath, AllClassPath)
    batch_loss =model.train_on_batch(inputs, targets)
    loss_graph = np.hstack((loss_graph, batch_loss))
    total_iterations += 1
    if total_iterations % 20 == 1:
        print ("Iteration=" + str(total_iterations) + " took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss))
print("Train Successful - Model saved")
save_model(model, model_path, weights_path)