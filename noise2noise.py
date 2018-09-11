
# coding: utf-8

# In[2]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']




# In[3]:


get_available_gpus()


# # Jupyter 
# 
# Python with good comments.
# 
# To run the below code you will need the follow dir structure:  
# + root  
# +-- outputs  
# +-- savedModel  
# +-- pnginputs: put your png inputs here  
# +-- targets: put your targets here  
# +-- inputs: put your exr gbuffer inputs here  
#     

# # Tensorflow nodes
# 
# Everything in tensorflow is a node, even scalars. You connect nodes to each other to make a graph and the flow of that graph is the runtime of the program.

# In[4]:


#Hello world
import tensorflow as tf

five = tf.constant([[5]],name="numbafive")
three = tf.constant([[3]],name="numbathree")
multiply = tf.matmul(five,three)

#Prints a tensor object
print(five)

# Start a sesssion, run the op
with tf.Session() as sess:
    print(sess.run(five)) #actually prints the content of the tensor
    print(sess.run(multiply))


# # Loading data
# 
# 
# ## "Variables" in Tensorflow
# 
# 3 kinds
# 
# **Placeholder**  
# A placeholder variable whose value is only assigned at runtime.  
# Used to store data.
# 
# **Variable**  
# A 'normal' variable that needs an initial value and that you can change its value throughout the course of your program.  
# Used to store weights, biases, anything that is trainable.  
# 
# 
# **Constant**  
# A variable that cannot be changed after it was declared and initialized.  
# Used to store anything that does not fall in the above, directory name lists, constant transformation matrices, constant scalars.
# 
# ## Creating Variables
# Without tensorflow: a = 12
# With tensorflow: tf_a = tf.get_variable(name="a", [1])
# 
# 
# get_variable() gets the variable if it is already defined in the scope
# tf.Variable() always initializes a new variable, giving it a numbered suffix in case the name already exists.
# 
# ## Dataset
# Tensorflow uses the 'dataset' object to manage datasets. The dataset is basically a queue and every call to it will return the next item in the queue. 
# 
# You can plug in tensorflow functions as lambda functions to transform the dataset using the map function from the dataset object. HOWEVER, you can only use tensorflow nodes to plug in as lambda functions. To use normal python, you need to wrap your python code in tf.py_func node. This is a special tensorflow node that compiles your python code into a tensorflow node, connectable to other nodes. 

# In[ ]:


#Neural Network demo code starts here
#Architecture based on SRRESNET from https://arxiv.org/pdf/1609.04802.pdf

import tensorflow as tf
BATCH_SIZE = 1

#some helper function
def preprocess(image):
        with tf.name_scope("preprocess"):
            # [0, 1] => [-1, 1]
            return image * 2 - 1
        
def deprocess(image):
        with tf.name_scope("deprocess"):
            # [-1, 1] => [0, 1]
            return (image + 1) / 2

        
def load_data(exr_input_dir,png_input_dir, target_dir, num_parallel_calls=1, batch_size=BATCH_SIZE, 	shuffle_buffer=0, num_epochs=40):
    import os
    import numpy as np
    import array
    
    
    def py_parse_exr(input_paths, target_paths, input_png_paths):
        #Anything in here needs to be pure python, no tensorflow
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(input_paths)
        HALF = Imath.PixelType(Imath.PixelType.FLOAT) 
        #if you want to re run this code, make sure you name the EXR channels accordingly: "normal" and "albedo"
        gbuffer = [array.array('f', exr_file.channel(Chan, HALF)).tolist() for Chan in ("normal.R", "normal.G", "normal.B") ]

        
        gbuffer = np.array(gbuffer, dtype = np.float32)
        gbuffer = np.swapaxes(gbuffer,0,1) #Shapes should almost be like typechecking, try to enforce them as often as possible. These are amongst some of the most common and hard to find bugs.
        return gbuffer, target_paths, input_png_paths
    


    def stack_buffers(gbuffer, target_paths, input_png_paths):
        target_img_file = tf.read_file(target_paths)
        target_img_tensor = tf.image.decode_png(target_img_file, channels=3)#uint8, 0 to 255
        target_img_tensor = preprocess(tf.image.convert_image_dtype(target_img_tensor,dtype=tf.float32))
        input_img_file = tf.read_file(input_png_paths)
        input_img_tensor = tf.image.decode_png(input_img_file, channels=3)#uint8, 0 to 255
        input_img_tensor = preprocess(tf.image.convert_image_dtype(input_img_tensor, dtype=tf.float32))

        gbuffer = tf.reshape(gbuffer, shape=[256,256,3])#EXRs usually come in [256x256,3]
        input_img_tensor = tf.concat([gbuffer,input_img_tensor],axis=2)
        return input_img_tensor, target_img_tensor
        

    #main
    input_gbuffer_paths = sorted([os.path.join(exr_input_dir,file) for file in os.listdir(exr_input_dir) if file.endswith("exr")])
    png_inputs_paths = sorted([os.path.join(png_input_dir,file) for file in os.listdir(png_input_dir) if file.endswith("png")])
    target_paths = sorted([os.path.join(target_dir,file) for file in os.listdir(target_dir) if file.endswith("png")])
    
    png_inputs = tf.constant(png_inputs_paths)
    targets = tf.constant(target_paths)
    input_gbuffer = tf.constant(input_gbuffer_paths)
    
    #transform the inputs from path strings to image tensors
    dataset = tf.data.Dataset.from_tensor_slices((input_gbuffer, targets, png_inputs))
    dataset = dataset.map(lambda input_file,target_paths,input_png_paths: tuple(tf.py_func(py_parse_exr, [input_file, target_paths, input_png_paths], [tf.float32,target_paths.dtype,input_png_paths.dtype])) )
    dataset = dataset.map(stack_buffers)
    
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    X,Y= iterator.get_next()
    X = tf.reshape(X, shape=[BATCH_SIZE,256,256,6])# Best practice to enforce the shapes, catches dim mismatches early
    Y = tf.reshape(Y, shape=[BATCH_SIZE,256,256,3])
    
    return X,Y
    


# # Scopes
# 
# Not necessary, but strongly encouraged for organization purposes.
# 
# These are ways to define scopes. In "normal" programming, scopes are implicit. In tensorflow, they are explicit, you make the scopes as you go and things never "scope" unless you tell them to.
# 
# ## Named scopes vs Variable Scopes
# 
# There are 2 types of scopes as defined by the title. The difference is as follows:
# 
# * tf.get_variable() ignores named scopes.
# 
# 
# 

# # Making the model
# 
# ## Graph programming (again).
# 
# Tensorflow's programming paradigm is called graph programming. Each assignment is a node in
# the tensorflow graph. To actually run the program, you need to explicitly call it to run by making a session and calling the run function of the session object on a node. This will that node and all it's dependencies. 
# 
# ## Canonical way of creating the neural network.
# 
# You take the input x and simply apply the different types of neural network layers. Each layer returns a node that you can continue connecting. It suffices to return the result of the last layer as an output.
# 
# ## A bit of background in how tensorflow trains the network.
# 
# Any "variable" is trainable and any time you declare a variable, it gets put in a "Collection" of trainable variables. The optimizer backpropagades on all variables under this "Collection". The code below hides the variable declaration unfortunately. They are declared when tf.layer.conv2d() or any other layer is declared. Remember, we use variables to represent the weights. 

# In[ ]:


#define the activation function to easily switch it
def activation_func(x):
    #return tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
    return tf.nn.relu(x)

def create_generator(x):
    def conv2d(x,kernel_size,num_feature_maps,stride):
        return tf.layers.conv2d(x, kernel_size=kernel_size, filters=num_feature_maps, strides=stride, padding='same', use_bias=False)

    def residual_block(X):
        with tf.variable_scope('residual_block'):

            skip = X
            output = X
            output = conv2d(output,3,64,1)
            output = tf.layers.batch_normalization(output, training=True)
            output = activation_func(output)
            output = conv2d(output,3,64,1)
            output = tf.layers.batch_normalization(output, training=True)
            output = output + skip
            return output

    def UpsamplingBlock(x):
        with tf.variable_scope('upsample_block'):
            x = tf.layers.conv2d(x, kernel_size=3, filters=256, strides=1, padding='same')
            x = tf.depth_to_space(x, 2)
            x = activation_func(x)
            return x
    
    
    with tf.variable_scope('generator'):
        output = x
        output = conv2d(output,9,64,1)
        output = activation_func(output)
        
        large_skip = output
        for i in range(16):
            with tf.variable_scope('residual_block' + str(i)):
                output = residual_block(output)
        


        output = conv2d(output,3,64,1)
        output = tf.layers.batch_normalization(output, training=True)
        output = output + large_skip

        output = conv2d(output,9,3,1)
        return output


# # Saving variables in tensorflow
# 
# Very easy, just one function call "saver.save()". Because all variables are under a collection, it is easy for tensorflow to save, load and manage those variables behind the scenes for you. 

# In[ ]:


def train(max_step, save_path="savedModel/model.ckpt", checkpoint=None):
    
    inputs, targets = load_data("./inputs","./pnginputs","./targets",num_epochs=max_step)
    outputs = create_generator(inputs)
    loss = tf.reduce_mean(tf.square(targets - outputs))
    
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    
    
    
    with tf.Session() as sess:
        if (checkpoint):
            saver.restore(sess, checkpoint)
            print("Restoring checkpoint")
        else:
            sess.run(init_op)
        
        for step in range(max_step):
            if((step+1)%2000 == 0):
                print("L1 Loss: ", sess.run(loss))
                save_path = saver.save(sess, save_path)
                print("Model saved in path: %s" % save_path)

            sess.run(train_op)
    


# # Restoring variables in tensorflow
# 
# ## 2 ways
# * The first way that is shown below. Tensorflow saves the values of the variables to disk, but you need to declare them if you want to use them again. In other words, you need to create your model exactly the way you created it before. However, this time, instead of initializing them, you load them from the checkpoint
# 
# * Tensorflow can save an additional file called the meta graph. This basically saves the information about the variables (and actually the whole tensorflow graph) themselves, not just their value. This means you couldn't need to call "create_generator" to declare the variables.

# In[ ]:


def test(load_path="savedModel/model.ckpt"):
    import os 

    inputs, targets = load_data("./inputs","./pnginputs","./targets",num_epochs=10000)
    outputs = create_generator(inputs)
    outputs = tf.image.convert_image_dtype(deprocess(outputs), dtype=tf.uint8, saturate=True)
    img_tensor = tf.map_fn(lambda x: tf.image.encode_png(x), outputs, dtype=tf.string, name="encode_png")
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init_op)
        # Restore variables from disk.
        saver.restore(sess, load_path)
        print("Model restored.")
        
        for idx in range(5):
            img_batches = sess.run(img_tensor)
            for img in img_batches:
                name = str(idx) + ".png"
                print("Writing: " + name)
                with open(os.path.join("./outputs",name), "wb") as f:
                    f.write(img)


# In[ ]:


if __name__ == "__main__":
    import sys
    #argument = sys.argv[1]
    #checkpoint_path = sys.argv[2]
    checkpoint_path = ""
    argument = "test"
    print(argument)
    if (argument == "train" or argument == ""):
        train(500000, save_path="savedModel/model.ckpt")
    elif(argument == "test"):
        test("savedModel/model.ckpt")
    elif (argument == "checkpointtrain"):
        train(500000, checkpoint=checkpoint_path)
    else:
        print("Invalid arguments")
        sys.exit(1)

