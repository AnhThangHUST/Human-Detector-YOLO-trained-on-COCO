import tensorflow as tf
import os
import cv2
import random
import time
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Parameters
raw_data_1='Rawdata'
data_path_1='Standard_data'
height_1=40
width_1=40
all_classes_1 = os.listdir(data_path_1)
number_of_classes_1 = len(all_classes_1)
color_channels_1=3
epochs_1=20
batch_size_1=16
batch_counter_1=0
model_save_name_1='Checkpoints3/case2/'


#model's unit definitions
class model_tools_1:
    # Defined functions for all the basic tensorflow components that we needed for building a model.
    # function definitions are in the respective comments

    def add_weights_1(self,shape_1):
        # a common method to create all sorts of weight connections
        # takes in shapes of previous and new layer as a list e.g. [2,10]
        # starts with random values of that shape.
        return tf.Variable(tf.truncated_normal(shape=shape_1, stddev=0.05))

    def add_biases_1(self,shape_1):
        # a common method to add create biases with default=0.05
        # takes in shape of the current layer e.g. x=10
        return tf.Variable(tf.constant(0.05, shape=shape_1))

    def conv_layer_1(self,layer_1, kernel_1, input_shape_1, output_shape_1, stride_size_1):
        #convolution occurs here.
        #create weights and biases for the given layer shape
        weights_1 = self.add_weights_1([kernel_1, kernel_1, input_shape_1, output_shape_1])
        biases_1 = self.add_biases_1([output_shape_1])
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1] mostly
        stride_1 = [1, stride_size_1, stride_size_1, 1]
        #does a convolution scan on the given image
        layer_1 = tf.nn.conv2d(layer_1, weights_1, strides=stride_1, padding='SAME') + biases_1
        return layer_1

    def pooling_layer_1(self,layer_1, kernel_size_1, stride_size_1):
        # basically it reduces the complexity involved by only taking the important features alone
        # many types of pooling is there.. average pooling, max pooling..
        # max pooling takes the maximum of the given kernel
        #kernel=[image_jump,rows,columns,depth]
        kernel_1 = [1, kernel_size_1, kernel_size_1, 1]
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,2,2,1] mostly
        stride_1 = [1, stride_size_1, stride_size_1, 1]
        return tf.nn.max_pool(layer_1, ksize=kernel_1, strides=stride_1, padding='SAME')

    def flattening_layer_1(self,layer_1):
        #make it single dimensional
        input_size_1 = layer_1.get_shape().as_list()
        new_size_1 = input_size_1[-1] * input_size_1[-2] * input_size_1[-3]
        return tf.reshape(layer_1, [-1, new_size_1]),new_size_1

    def fully_connected_layer_1(self,layer_1, input_shape_1, output_shape_1):
        #create weights and biases for the given layer shape
        weights_1 = self.add_weights_1([input_shape_1, output_shape_1])
        biases_1 = self.add_biases_1([output_shape_1])
        #most important operation
        layer_1 = tf.matmul(layer_1,weights_1) + biases_1  # mX+b
        return layer_1

    def activation_layer_1(self,layer_1):
        # we use Rectified linear unit Relu. it's the standard activation layer used.
        # there are also other layer like sigmoid,tanh..etc. but relu is more efficent.
        # function: 0 if x<0 else x.
        return tf.nn.relu(layer_1)
    pass


#tools for image processing and data handing.
class utils_1:
    image_count_1 = []
    count_buffer_1=[]
    class_buffer_1=all_classes_1[:]
    def __init__(self):
        self.image_count_1 = []
        self.count_buffer_1 = []
        for i in os.walk(data_path_1):
            if len(i[2]):
                self.image_count_1.append(len(i[2]))
        self.count_buffer_1=self.image_count_1[:]

    # processing images into arrays and dispatch as batches whenever called.
    def batch_dispatch_1(self,batch_size_1=batch_size_1):
        global batch_counter_1
        if sum(self.count_buffer_1):

            class_name_1 = random.choice(self.class_buffer_1)
            choice_index_1 = all_classes_1.index(class_name_1)
            choice_count_1 = self.count_buffer_1[choice_index_1]
            if choice_count_1==0:
                class_name_1=all_classes_1[self.count_buffer_1.index(max(self.count_buffer_1))]
                choice_index_1 = all_classes_1.index(class_name_1)
                choice_count_1 = self.count_buffer_1[choice_index_1]

            slicer_1=batch_size_1 if batch_size_1<choice_count_1 else choice_count_1
            img_ind_1=self.image_count_1[choice_index_1]-choice_count_1
            indices_1=[img_ind_1,img_ind_1+slicer_1]
            images_1 = self.generate_images_1(class_name_1,indices_1)
            labels_1 = self.generate_labels_1(class_name_1,slicer_1)

            self.count_buffer_1[choice_index_1]=self.count_buffer_1[choice_index_1]-slicer_1
        else:
            images_1,labels_1=(None,)*2
        return images_1, labels_1

    #gives one hot for the respective labels
    def generate_labels_1(self,class_name_1,number_of_samples_1):
        one_hot_labels_1=[0]*number_of_classes_1
        one_hot_labels_1[all_classes_1.index(class_name_1)]=1
        one_hot_labels_1=[one_hot_labels_1]*number_of_samples_1
        #one_hot_labels=tf.one_hot(indices=[all_classes.index(class_name)]*number_of_samples,depth=number_of_classes)
        return one_hot_labels_1

    # image operations
    def generate_images_1(self,class_name_1,indices_1):
        batch_images_1=[]
        choice_folder_1=os.path.join(data_path_1,class_name_1)
        selected_images_1=os.listdir(choice_folder_1)[indices_1[0]:indices_1[1]]
        for image_1 in selected_images_1:
            img_1=cv2.imread(os.path.join(choice_folder_1,image_1))
            batch_images_1.append(img_1)
        return batch_images_1

#generating our own model, explanations are given respectively
def generate_model_1(images_ph_1,number_of_classes_1):
    #MODEL ARCHITECTURE:
    #level 1 convolution
    network_1=model_1.conv_layer_1(images_ph_1,5,3,16,1)
    network_1=model_1.pooling_layer_1(network_1,2,2)
    network_1=model_1.activation_layer_1(network_1)
    print(network_1)
    #level 2 convolution
    network_1=model_1.conv_layer_1(network_1,5,16,32,1)
    network_1=model_1.pooling_layer_1(network_1,2,2)
    network_1=model_1.activation_layer_1(network_1)
    print(network_1)
    # level 3 convolution
    network_1 = model_1.conv_layer_1(network_1, 4, 32, 32, 1)
    network_1 = model_1.pooling_layer_1(network_1, 2, 2)
    network_1 = model_1.activation_layer_1(network_1)
    print(network_1)
    #flattening layer
    network_1,features_1=model_1.flattening_layer_1(network_1)
    print(network_1)
    #fully connected layer
    network_1=model_1.fully_connected_layer_1(network_1,features_1,64)
    network_1=model_1.activation_layer_1(network_1)
    print(network_1)

    #output layer
    network_1=model_1.fully_connected_layer_1(network_1,64,number_of_classes_1)
    print(network_1)

    return network_1

#training happens here
def trainer_1(network_1,number_of_images_1):
    #find error like squared error but better
    cross_entropy_1=tf.nn.softmax_cross_entropy_with_logits_v2(logits=network_1,labels=labels_ph_1)

    #now minize the above error
    #calculate the total mean of all the errors from all the nodes
    cost_1=tf.reduce_mean(cross_entropy_1)
    tf.summary.scalar("cost", cost_1)#for tensorboard visualisation

    #Now backpropagate to minimise the cost in the network.
    optimizer_1=tf.train.AdamOptimizer(0.0005).minimize(cost_1)
    tf.add_to_collection("activation", network_1)
    #Start the session
    session_1.run(tf.global_variables_initializer())
    writer_1 = tf.summary.FileWriter(model_save_name_1, graph=tf.get_default_graph())
    merged_1 = tf.summary.merge_all()
    saver_1 = tf.train.Saver(max_to_keep=None)
    counter_1=0
    for epoch_1 in range(epochs_1):
        tools_1 = utils_1()
        for batch_1 in range(int(number_of_images_1 / batch_size_1)):
            counter_1+=1
            images_1, labels_1 = tools_1.batch_dispatch_1()
            if images_1 == None:
                break
            loss_1,summary_1 = session_1.run([cost_1,merged_1], feed_dict={images_ph_1: images_1, labels_ph_1: labels_1})
            print('loss', loss_1)
            session_1.run(optimizer_1, feed_dict={images_ph_1: images_1, labels_ph_1: labels_1})

            print('Epoch number ', epoch_1, 'batch', batch_1, 'complete')
            writer_1.add_summary(summary_1,counter_1)
            if counter_1 % 100 == 0:
                saver_1.save(session_1, model_save_name_1,global_step= counter_1)
    saver_1.save(session_1,"Checkpoints/case_1/CNN_MHI")

session_1=tf.Session()

#Input and Output for CNN_1
images_ph_1=tf.placeholder(tf.float32,shape=[None,height_1,width_1,color_channels_1])
labels_ph_1=tf.placeholder(tf.float32,shape=[None,number_of_classes_1])
tools_1=utils_1()
model_1=model_tools_1()
network_1=generate_model_1(images_ph_1,number_of_classes_1)
number_of_images_1 = sum([len(files) for r, d, files in os.walk("Standard_data")])
trainer_1(network_1,number_of_images_1)
