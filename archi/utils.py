'''
Created on Feb 4, 2019

@author: annst
'''
import numpy as np
import tensorflow as tf
import json
import imageio

def load_graph(graph_def, frozen_graph_filename,prefix, input_map=None,return_elements=None):
    """Load frozen graph into default graph
    prefix - name prefix of frozen graph
    """
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Then, we import the graph_def into a new Graph and returns it 
    #with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
    output = tf.import_graph_def(graph_def, name=prefix,input_map=input_map, return_elements=return_elements )
    return output

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img

def hist_summary(w,b,out,name):
    """
    Create histogram summary for a layer and its output.
    Args:
        w - weights
        b - bias
        out - activations
        name - name of the layer
    Returns:
        None
    """
    tf.summary.histogram("{}/weights".format(name),w)
    tf.summary.histogram("{}/activation".format(name),out)
    tf.summary.histogram("{}/bias".format(name),b)
    
def dict_to_json(fpath,dict_values): 
    """Save dictionary in json file
    Args:
        fpath  - path of intended json file, should have .json extension at the end
        dict_values - dictionary that you want to save"""
    with open(fpath, 'w') as fp:
        json.dump(dict_values, fp)

def json_to_dict(fpath):
    """Restore dictionary from json file"""
    with open(fpath,'r') as fp:
        dict = json.load(fp)
    return dict

def save_images_gan(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imageio.imwrite(save_path, img)