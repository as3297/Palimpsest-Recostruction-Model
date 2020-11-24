'''
Created on Jun 27, 2019

@author: annst
'''
import os
import tensorflow as tf
import numpy as np
#import input_data
from utils import *
#from ops import *
from skimage import io, filters, transform,color
from skimage import util as util_sklearn
from matplotlib import pyplot as plt
import imageio
import random
import re

bands = ("LED365_01_corr","LED445_01_raw","LED470_01_raw","LED505_01_raw","LED530_01_raw","LED570_01_raw",\
             "LED617_01_raw","LED625_01_raw","LED700_01_raw","LED735_01_raw","LED870_01_raw")

class BSSModel():
    def __init__(self,log_dir,image_size,dataset,data_dir,timestep):
        self.image_size = image_size
        self.log_dir = os.path.join(log_dir,timestep)
        self.log_dir_results = os.path.join(self.log_dir,'results',dataset)
        self.log_dir_latent = os.path.join(self.log_dir,'latent')
        self.log_dir_inverse = os.path.join(self.log_dir,'inverse')
        self.log_dir_eval = os.path.join(self.log_dir_results,'evaluate')
        self.data_dir = data_dir
    
    def threshold_ims(self,ims,thresh_value):
        """Threshold image to extract overtext
        thresh_value - normalized brightness value"""
        thresh_ims = np.zeros((ims.shape))
        for i in range(len(ims)):
            thresh_ims[i] = ims[i]>thresh_value#>0.32 #>-0.35# filters.threshold_otsu(ims[i])
            plt.figure()
            plt.hist(ims[i].ravel())
            #plt.show()
        return thresh_ims
    
    def prepare_log_dirs(self,im_list,nb_obs=1):
        if not os.path.isdir(os.path.join(self.log_dir_inverse)):
            os.makedirs(os.path.join(self.log_dir_inverse))
        #create folder for validation results
        if not os.path.isdir(os.path.join(self.log_dir_results,"evaluate")):
            os.makedirs(os.path.join(self.log_dir_results,"evaluate"))

        characters= list(set([os.path.normpath(impath).split(os.sep)[-2] for impath in im_list]))
        for digit in characters:
            if not os.path.isdir(os.path.join(self.log_dir_results,"under_pred",str(digit))):
                os.makedirs(os.path.join(self.log_dir_results,"under_pred",str(digit)))
            if not os.path.isdir(os.path.join(self.log_dir_results,"overlap_pred",str(digit))):
                os.makedirs(os.path.join(self.log_dir_results,"overlap_pred",str(digit)))
            if not os.path.isdir(os.path.join(self.log_dir_results,"background_pred",str(digit))):
                os.makedirs(os.path.join(self.log_dir_results,"background_pred",str(digit)))
        if nb_obs>1:
            for im in im_list:
                fname = os.path.normpath(im).split(os.sep)[-1]
                char = os.path.normpath(im).split(os.sep)[-2]
                if not os.path.isdir(os.path.join(self.log_dir_results, "overlap_pred", char,fname)):
                    os.makedirs(os.path.join(self.log_dir_results, "overlap_pred", char,fname))
                    
    def save_ims_eval_results(self,gt,gen_undertext,gen_overtext,gen_overlap,epoch,impath):
        split_path= os.path.normpath(impath).split(os.sep)
        char,fname = split_path[-3], split_path[-1]
        imageio.imwrite(os.path.join(self.log_dir_eval,char,"base_"+fname[:-4]+"_"+str(epoch)+".jpg"),gt.reshape(self.image_size,self.image_size))
        imageio.imwrite(os.path.join(self.log_dir_eval,char,"under_"+fname[:-4]+"_"+str(epoch)+".jpg"),gen_undertext.reshape(self.image_size,self.image_size))
        imageio.imwrite(os.path.join( self.log_dir_eval,char,"over_"+fname[:-4]+"_"+str(epoch)+".jpg"), gen_overtext[:,:,0])
        imageio.imwrite(os.path.join(self.log_dir_eval,char,"overlap_"+fname[:-4]+"_"+str(epoch)+".jpg"),gen_overlap.reshape(self.image_size,self.image_size))
        
    def save_predicted_ims(self,im_list,under_ims,overlap_ims,background_ims=None,nb_obs=1):
        for i in range(len(im_list)):
            split_path= os.path.normpath(im_list[i]).split(os.sep)
            digit,fname = split_path[-2], split_path[-1]
            if nb_obs>1:
                imageio.imwrite(os.path.join(self.log_dir_results, "under_pred", digit, fname)+".png", under_ims[i])
                for obs_idx in range(nb_obs):
                    imageio.imwrite(os.path.join(self.log_dir_results,"overlap_pred",digit,fname,bands[obs_idx]+"_"+str(obs_idx))+".png",overlap_ims[i][:,:,obs_idx])
                if background_ims!=None:
                    imageio.imwrite(os.path.join(self.log_dir_results, "background_pred", digit, fname)+".png", background_ims[i])
            else:
                imageio.imwrite(os.path.join(self.log_dir_results, "under_pred", digit, fname), under_ims[i])
                if background_ims!=None:
                    imageio.imwrite(os.path.join(self.log_dir_results, "background_pred", digit, fname), background_ims[i])


def read_batch(im_list,image_size,data_dir,nb_obs=1):
    """
    Read original and overtext image batch from Archimedes datasets
    im_list - list of files to read
    image_size - image size
    data_dir - dataset diectory
    nb_obs - number of channels, or number of bands

    """
    batch_size = len(im_list)
    if nb_obs>1:
        ims_batch = np.zeros((batch_size,image_size,image_size,nb_obs),dtype="float32")
    else:
        ims_batch = np.zeros((batch_size, image_size, image_size,1),dtype="float32")
    ims_over = np.zeros((batch_size,image_size,image_size,1))
    for i in range(batch_size):
        im_path_org = re.sub("~",data_dir,im_list[i])
        im_path_over = re.sub("~",data_dir+"_thresholded",im_list[i])
        if nb_obs>1:
            for obs_idx in range(nb_obs):
                im_raw = io.imread(fname=os.path.join(im_path_org,bands[obs_idx],im_list[i].split(os.sep)[-1]+".tif"))
                if np.max(im_raw)>1:
                    im_raw = ((im_raw/65532)*255).astype("uint8")#scaling of one band
                im_raw = color.rgb2gray(im_raw)
                im_raw = transform.resize(im_raw,(image_size,image_size),preserve_range=True)
                print(np.max(im_raw))
                ims_batch[i,:,:,obs_idx] = 2.0 * im_raw - 1
        else:
            im_raw = io.imread(fname=os.path.join(im_path_org), as_gray=True)
            im_norm = im_raw
            im_scaled = im_norm.reshape(image_size,image_size,1)
            ims_batch[i] = 2.0 * im_scaled - 1
        if nb_obs>1:
            ims_over[i] = io.imread(fname=im_path_over+".bmp", as_gray=True).reshape(image_size, image_size, 1)
        else:
            ims_over[i] = io.imread(fname=im_path_over, as_gray=True).reshape(image_size,image_size,1)
            ims_over[i] = 2.0*(1 - ims_over[i]/np.max(ims_over[i])) - 1.0
            ims_over[i][ims_over[i]<0.9]=-1


    assert np.max(ims_batch) <= 1 and np.min(ims_batch) >= -1, "Text min {}, max {}".format(np.min(ims_batch),
                                                                                                  np.max(ims_batch))
    assert np.max(ims_over) <= 1 and np.min(ims_over) >= -1, "Over Text min {}, max {}".format(
        np.min(ims_over), np.max(ims_over))
    return ims_batch, ims_over

def make_im_list(data_dir):
    im_list = []
    for charac in os.listdir(data_dir):
        flist_char = os.listdir(os.path.join(data_dir,str(charac)))
        for fname in flist_char:
            im_list.append(os.path.join("~",str(charac),fname))
    return im_list


if __name__ == "__main__":
    datadir = r"C:\\Data\\PhD\\bss_gan_palimpsest\\datasets\\Archimedes\\test_obscured"
    imlist = make_im_list(datadir)
    ims, overtext = read_batch(imlist,data_dir=datadir,image_size=64, nb_obs=1)
    fig, ax = plt.subplots(nrows=1,ncols=2)
    print(np.max(ims[0,:,:,0]))
    for i in range(1):
        ax[i].imshow(ims[0,:,:,i],cmap="gray")
    plt.show()