'''
Created on Jun 26, 2019

@author: annst
'''
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import tensorflow as tf
import numpy as np
#from ops import *
from skimage import io, filters, transform
from skimage import util as util_sklearn
from matplotlib import pyplot as plt
from scipy.optimize import fmin
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import re
from undertext_gen_train import Text_Back_Gen



class LatentInvModel():
    def __init__(self,z_utb,z_b,image_size,graph,mask,gen_restore_path_utb,gen_restore_path_b,data_dir,\
                 ganloss,batch_size,mixing_model,model_idx=None,nb_obs=1, with_background=True,restore_mix=False,l1_mult=0.01,l2_mult=1.0,nonexc_mult=0.0, cosine_mult=0.0,gmm_mult=0):

        self.l1_mult = l1_mult
        self.l2_mult = l2_mult
        self.gmm_mult = gmm_mult
        self.nonexc_mult = nonexc_mult
        self.cosine_mult = cosine_mult
        self.z_utb = z_utb
        self.batch_size = batch_size
        self.image_size = image_size
        self.mask = mask
        self.gen_restore_path_utb = gen_restore_path_utb
        self.gen_restore_path_b = gen_restore_path_b
        self.graph = graph
        self.model_idx = model_idx
        self.nb_obs=nb_obs
        self.z_b = z_b
        self.global_step_dict = {}
        self.data_dir = data_dir
        self.ganloss = ganloss
        self.text_back_model = Text_Back_Gen(batch_size=batch_size,graph=graph)
        self.latent_models_dict={}
        self.optimizers_under = {}
        self.optimizers_back = {}
        self.with_background = with_background
        self.restore_mix = restore_mix
        if "3d" in mixing_model:
            with self.graph.as_default() as graph_def:
                self.text_back_model.compile_mixing_net_3d()
        self.mixing_net = eval("self.text_back_model." + mixing_model)
            
    
    def build_model_1_latent_v(self,lr_under,lr_back,ims_idx,latent_init_under,latent_init_back,init_idx,gpu_idx,im,overtext_im,nb_classes=None):
        with self.graph.as_default() as graph_def:
           
            #with tf.device("/device:GPU:{}".format(gpu_idx)):
            with tf.variable_scope("img/{}/{}".format(ims_idx,init_idx)):
                overtext = tf.constant(overtext_im,tf.float32,name="overtext_lat")
                org = tf.constant(im,tf.float32,name="org_lat")
                latent_variables_undertext = self.latent_variable(ims_idx, init_idx, latent_init_under, self.z_utb, "utb",
                                                                  nb_classes=nb_classes)
                latent_variables_background = self.latent_variable(ims_idx, init_idx, latent_init_back, self.z_b, "ut",
                                                                   nb_classes=nb_classes)
            undertext = self.text_back_model.gen_underText(latent_variables_undertext)

            background = self.text_back_model.gen_background(latent_variables_background)
            if self.with_background:
                est = self.mixing_net(undertext,background,overtext,nb_obs=self.nb_obs)
            else :
                est = self.mixing_net(undertext, overtext,nb_obs=self.nb_obs)
            if self.mask>0:
                overtext_mask = self.build_mask(overtext)
            else:
                overtext_mask=None
            if self.ganloss!=0:
                cost_l2 = self.l2_mult * tf.reduce_mean(self.l2_loss(pred=est, org=org,mask=overtext_mask,alpha=self.mask))+self.ganloss*tf.reduce_mean(self.text_back_model.disc_utb(undertext))
            else:
                cost_l2 = self.l2_mult * tf.reduce_mean(self.l2_loss(pred=est, org=org,mask=overtext_mask,alpha=self.mask))
            cost_l1 = self.l1_mult * tf.reduce_mean(self.l1_loss(pred=est, org=org, mask=overtext_mask, alpha=self.mask))
            if self.nb_obs>1:
                cosine_loss = self.cosine_mult * tf.reduce_mean(self.cosine_loss(pred=est,org=org,mask=overtext_mask,alpha=self.mask))
            else:
                cosine_loss = 0.0
            cost_back = cost_l1+cost_l2+cosine_loss
            cost_under = cost_l1+cost_l2+cosine_loss
            with tf.variable_scope("img/{}/{}/optimize_lat/utb".format(ims_idx,init_idx)):
                optimizer_under = tf.train.AdamOptimizer(lr_under).minimize(cost_under,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="img/{}/{}/latent_variables/utb".format(ims_idx,init_idx)))
            with tf.variable_scope("img/{}/{}/optimize_lat/ut".format(ims_idx, init_idx)):
                optimizer_back = tf.train.AdamOptimizer(lr_back).minimize(cost_back, var_list=tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope="img/{}/{}/latent_variables/ut".format(ims_idx, init_idx)))
            self.optimizers_under[(ims_idx,init_idx)] = optimizer_under
            self.optimizers_back[(ims_idx, init_idx)] = optimizer_back
            summ_est = tf.summary.image("overlapped_{}_{}".format(ims_idx,init_idx), tf.slice(est, [0, 0, 0, 0], [-1, self.image_size, self.image_size,
                                                                                                                      min(
                                                                                                                          self.nb_obs,
                                                                                                                          3 if self.nb_obs > 1 else 1)]))
            summ_under = tf.summary.image("underterxt_{}_{}".format(ims_idx, init_idx), undertext)
            summ = tf.summary.merge([summ_est,summ_under])
            self.latent_models_dict[(ims_idx,init_idx)]={"cost":cost_under,"latent_variables_undertext":latent_variables_undertext,\
                "latent_variables_background":latent_variables_background,"undertext":undertext,"background":background,"est":est,"overtext_plh":overtext,"org_plh":org,"summ":summ}

    def build_mask(self,overtext):
        with tf.variable_scope("CreateMask", reuse=tf.AUTO_REUSE):
             if "mnist" in self.data_dir.lower():
                 overtext_mask = 1 - overtext
             elif "archi" in self.data_dir.lower():
                 overtext_mask = 1-(overtext + 1) / 2
             return overtext_mask

    def build_model_mixmodel(self,lr_rate):
        with self.graph.as_default() as graph_def:
            overtext = tf.placeholder(tf.float32, shape=(None, self.image_size,self.image_size,1))
            org = tf.placeholder(tf.float32, shape=(None, self.image_size,self.image_size,self.nb_obs))
            latent_variables_undertext = tf.placeholder(tf.float32, shape=(None, self.z_utb))
            latent_variables_background = tf.placeholder(tf.float32, shape=(None, self.z_b))
            undertext = self.text_back_model.gen_underText(latent_variables_undertext)
            summ_under = tf.summary.image("underterxt",undertext)
            summ_over = tf.summary.image("overtext",overtext)
            background = self.text_back_model.gen_background(latent_variables_background)

            if self.with_background:
                background_for_summary =self.mixing_net(tf.zeros_like(background) - 1.0,
                                                                         background,
                                                                         tf.zeros_like(background) - 1.0, nb_obs=self.nb_obs)

                est = self.mixing_net(undertext,background,overtext,nb_obs=self.nb_obs)
            else:
                est =self.mixing_net(undertext, overtext,nb_obs=self.nb_obs)

            summ_back = tf.summary.image("background", tf.slice(background_for_summary, [0, 0, 0, 0],
                                                                [min(self.batch_size,3), self.image_size, self.image_size,
                                                                 min(self.nb_obs, 3)]))
            summ_est = tf.summary.image("overlapped",tf.slice(est,[0,0,0,0],[min(self.batch_size,3),self.image_size,self.image_size,min(self.nb_obs,3 if self.nb_obs>1 else 1)]))
            summ_org = tf.summary.image("original",tf.slice(org,[0,0,0,0],[min(self.batch_size,3),self.image_size,self.image_size,min(self.nb_obs,3 if self.nb_obs>1 else 1)]))
            if self.mask>0:
                overtext_mask = self.build_mask(overtext)
                summ_mask = tf.summary.image("mask",overtext_mask)
            else:
                overtext_mask=None
            if self.ganloss!=0:
                l2_cost = self.l2_mult * tf.reduce_mean(self.l2_loss(pred=est, org=org,mask=overtext_mask,alpha=self.mask))+self.ganloss*tf.reduce_mean(self.discriminator(undertext))
            else:
                l2_cost = self.l2_mult * tf.reduce_mean(self.l2_loss(pred=est, org=org,mask=overtext_mask,alpha=self.mask))
            l1_cost = self.l1_mult * tf.reduce_mean(self.l1_loss(pred=est, org=org, mask=overtext_mask, alpha=self.mask))
            bare_undertext = tf.reduce_mean(self.mixing_net(undertext,-1*tf.ones_like(undertext),-1*tf.ones_like(undertext),nb_obs=self.nb_obs),axis=3,keepdims=True)
            non_exc_loss = self.nonexc_mult * tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.image.sobel_edges(bare_undertext),tf.image.sobel_edges(-undertext)))
            if self.nb_obs>1:
                cosine_loss = self.cosine_mult * tf.reduce_mean(self.cosine_loss(pred=est,org=org,mask=overtext_mask,alpha=self.mask))
            else:
                cosine_loss = 0.0
            cost = l1_cost+l2_cost+non_exc_loss+cosine_loss
            cost_summ_cos = tf.summary.scalar("cosine_sim_loss",cosine_loss)
            cost_summ_l1 = tf.summary.scalar("Mix_loss_l1",l1_cost)
            cost_summ_l2 = tf.summary.scalar("Mix_loss_l2",l2_cost)
            cost_summ_nonexc = tf.summary.scalar("Exclusion loss", non_exc_loss)
            summ_undertext_after_mixing = tf.summary.image("underterxt_bare",bare_undertext)

            optimizer_mix = tf.train.AdamOptimizer(lr_rate).minimize(cost, var_list=tf.get_collection(
                      tf.GraphKeys.GLOBAL_VARIABLES, scope="MixingNet"))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer_mix = tf.group([optimizer_mix, update_ops])
            summary = tf.summary.merge([summ_under,summ_org,summ_est,summ_back,summ_over,summ_mask,\
                                        cost_summ_l1,cost_summ_l2,cost_summ_nonexc,\
                                        summ_undertext_after_mixing,cost_summ_cos])
            return optimizer_mix,cost,latent_variables_undertext,latent_variables_background,org,overtext,undertext,est,summary

    def l1_loss(self,pred,org,mask=None,alpha=0.0):
        """
        mask - overtext mask, shape [batch,channel,width,height]
        alpha - value of overtext
        """
        if mask!=None:
            mask = tf.clip_by_value(t=mask,clip_value_min=alpha,clip_value_max=1,)
            recon_loss = tf.reduce_sum(tf.multiply(mask,tf.abs(pred - org)),axis=[1,2,3])
            
        else:
            recon_loss = tf.reduce_sum(tf.abs(pred - org),axis=[1,2,3])
        return recon_loss

    def l2_loss(self, pred, org, mask=None, alpha=0.0):
        """
        mask - overtext mask, shape [batch,channel,width,height]
        alpha - value of overtext
        """
        if mask != None:
            mask = tf.clip_by_value(t=mask, clip_value_min=alpha, clip_value_max=1, )
            recon_loss = tf.reduce_sum(tf.multiply(mask, (pred - org)**2), axis=[1, 2, 3])

        else:
            recon_loss = tf.reduce_sum((pred - org)**2, axis=[1, 2, 3])
        return recon_loss

    def cosine_loss(self, pred, org, mask=None, alpha=0.0):
        """
        mask - overtext mask, shape [batch,channel,width,height]
        alpha - value of overtext
        """
        pred = tf.transpose(pred, (3,0,1,2))
        pred = tf.reshape(pred, (self.nb_obs, -1))
        org = tf.transpose(org, (3, 0, 1, 2))
        org= tf.reshape(org, (self.nb_obs, -1))
        def cosine_similarity(a,b):
            a_norm2 = tf.nn.l2_normalize(a ,0)
            b_norm2 = tf.nn.l2_normalize(b, 0)
            print(a_norm2.get_shape())
            ab_norm2 = tf.multiply(a_norm2,b_norm2)
            return 1-ab_norm2

        if mask != None:
            mask = tf.clip_by_value(t=mask, clip_value_min=alpha, clip_value_max=1, )
            mask = tf.reshape(mask,[-1])
            recon_loss = tf.multiply(cosine_similarity(org,pred),mask)
        else:
            recon_loss = cosine_similarity(org,pred)
        recon_loss = tf.reduce_sum(tf.reshape(recon_loss,[-1,self.image_size,self.image_size]),axis=[1,2])
        return recon_loss

    def binary_cross_entropy(self,org_images,pred_images):
        """Binary cross entropy loss
        Args:
        org_images - original images flatten
        pred_image - predicted images of shape [batch_size,height,width,channels]"""
        re = -1.0*tf.reduce_sum((org_images*tf.log(1e-10 + pred_images)+(1-org_images)*tf.log(1e-10 +1-pred_images)),axis=[1,2,3])
        return re
    def latent_variable(self,ims_idx,init_idx,latent_init,nb_z,mode,nb_classes=None):
        with tf.variable_scope("latent_variables/{}".format(mode)):
            mean_reconst_under = tf.get_variable("z_v",shape=[1,nb_z], initializer = tf.constant_initializer(latent_init,dtype=tf.float32), constraint=lambda x: tf.clip_by_value(x, -1., 1.))
        return mean_reconst_under
        
    def create_session(self):
        config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=False)
        #config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        
    def restore_generator(self):
        vars_utb = self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBinary")
        saver_restore = tf.train.Saver(vars_utb)
        saver_restore.restore(self.sess, self.gen_restore_path_utb)
        vars_b = self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack")
        saver_restore = tf.train.Saver(vars_b)
        saver_restore.restore(self.sess, self.gen_restore_path_b)
        if self.restore_mix:
            vars_mix= self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES, scope="MixingNet")
            saver_restore = tf.train.Saver(vars_mix)
            saver_restore.restore(self.sess, self.gen_restore_path_b)
        
    def init_model(self):
        with self.graph.as_default() as graph_def:
            self.sess.run(tf.global_variables_initializer())
            
    
    def create_model(self):
        self.create_session()
        self.init_model()
        self.restore_generator()
        var_store_mix=self.graph.get_collection(name=tf.GraphKeys.GLOBAL_VARIABLES,scope="MixingNet")
        self.saver = tf.train.Saver(var_store_mix,max_to_keep=2)


