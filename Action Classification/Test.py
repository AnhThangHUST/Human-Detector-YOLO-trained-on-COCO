import cv2
import tensorflow as tf
import os
import glob
import numpy as np
import csv
import pandas as pd
idx =0
fri = 0
font = cv2.FONT_HERSHEY_SIMPLEX
labels_1 = np.zeros((1, 2))
height = 40
width = 40
#-------------------------------------------------------------------------------------------------------------------

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)

            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:
            self.activation = tf.nn.softmax(tf.get_collection('activation')[0])
              # BY NAME:
            #self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data, label_ph):
        """ Running the activation operation previously imported """
        return self.sess.run(self.activation, feed_dict={"Placeholder:0": data, "Placeholder_1:0": label_ph})


IDX = []

TP_result_Abnormal = []
TP_result_Normal = []

FP_result_Abnormal = []
FP_result_Normal = []
TP_result_Abnormal_rate = []
TP_result_Normal_rate = []
FP_result_Abnormal_rate = []
FP_result_Normal_rate = []

# Run all saved models to calculate TP, FP and FN results
for index in range(1,53):
    model_path = 'Checkpoints3/case2/-' + str(100*index)
    model_1 = ImportGraph(model_path)

    img_dir_MHI_Abnormal = "Test/Abnormal"
    img_dir_MHI_Normal = "Test/Normal"

    Abnormal_data = len(os.listdir(img_dir_MHI_Abnormal))
    Normal_data = len(os.listdir(img_dir_MHI_Normal))


    data_path_Abnormal = os.path.join(img_dir_MHI_Abnormal)
    data_path_Normal = os.path.join(img_dir_MHI_Normal)

    files_Abnormal = glob.glob(data_path_Abnormal+'/*.*')
    files_Normal = glob.glob(data_path_Normal+'/*.*')


    TP_Abnormal = 0
    FP_Abnormal = 0
    TP_Normal = 0
    FP_Normal = 0
    for f1 in files_Abnormal:
        img = cv2.imread(f1)
        img = cv2.resize(img, (width, height))
        img = img.reshape(1, width, height, 3)
        result1 = model_1.run(img,labels_1)
        if (result1[0][0] ==np.amax(result1)):
            TP_Abnormal = TP_Abnormal + 1
        else:

            FP_Abnormal = FP_Abnormal+1

    TP_result_Abnormal.append(TP_Abnormal)
    FP_result_Abnormal.append(FP_Abnormal)

    for f1 in files_Normal:
        img = cv2.imread(f1)
        img = cv2.resize(img, (width, height))
        img = img.reshape(1, width, height, 3)
        result1 = model_1.run(img, labels_1)
        if (result1[0][0] == np.amax(result1)):
            FP_Normal = FP_Normal + 1
        else:

            TP_Normal = TP_Normal + 1


    TP_result_Normal.append(TP_Normal)
    FP_result_Normal.append(FP_Normal)


    #Tinh do chinh xac cua tung class


    TP_Abnormal_rate = TP_Abnormal/Abnormal_data*100
    TP_result_Abnormal_rate.append(TP_Abnormal_rate)
    TP_Normal_rate = TP_Normal/Normal_data*100
    TP_result_Normal_rate.append(TP_Normal_rate)

    FP_Abnormal_rate = FP_Abnormal/Abnormal_data*100
    FP_result_Abnormal_rate.append(FP_Abnormal_rate)
    FP_Normal_rate = FP_Normal/Normal_data*100
    FP_result_Normal_rate.append(FP_Normal_rate)

# print(TP_result_Abnormal)
# print(FP_result_Normal)
# print(TP_result_Normal)
# print(FP_result_Normal)
# print(TP_result_Normal_rate)
# print(FP_result_Abnormal_rate)
# print(TP_result_Abnormal_rate)
# print(FP_result_Normal_rate)
# print(index)
IDX.append(index)
#Save all calculated results to a xlsx file

data ={'TP_result_Abnormal':TP_result_Abnormal,'FP_result_Abnormal':FP_result_Abnormal,'TP_result_Normal':TP_result_Normal,'FP_result_Normal':FP_result_Normal,
       'Ab_rate':TP_result_Abnormal_rate,'Nor_rate':TP_result_Normal_rate,'Ab-Nor':FP_result_Abnormal_rate,'Nor-Ab':FP_result_Normal_rate}
df = pd.DataFrame(data)
df.to_excel('Result_TP_FP_case2-3_lr=0,0005.xlsx', header=True, index=True)

