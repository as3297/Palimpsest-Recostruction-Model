'''
Created on Apr 22, 2019

@author: annst
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import numpy as np
from skimage import io, filters, transform
from skimage import util as util_sklearn
from matplotlib import pyplot as plt
import time
import random
import re
import importlib
import argparse
from tensorflow.python.client import timeline
from finv_util import LatentInvModel
import json
from dataset_util import *
from time import gmtime, strftime
FLAGS={}


class BSSModelTraining(BSSModel):
    def __init__(self,log_dir,gen_restore_path_utb,gen_restore_path_b,image_size,dataset,\
                 mask,nb_utb,nb_b,nb_classes,model_name,data_dir,ganloss,in_iter,out_iter,\
                 with_background,restore_mix, lr_mix, lr_under,lr_back,mixing_model,timestep, \
                 l1_mult, l2_mult, nonexc_mult,cosine_mult):
        super().__init__(log_dir,image_size, dataset,data_dir,timestep)
        self.mask=mask
        self.z_utb=nb_utb
        self.z_b=nb_b
        self.nb_classes=nb_classes
        self.model_name=model_name
        self.ganloss=ganloss
        self.in_iter=in_iter
        self.out_iter=out_iter
        self.gen_restore_path_utb=gen_restore_path_utb
        self.gen_restore_path_b=gen_restore_path_b
        self.with_background = with_background
        self.restore_mix = restore_mix
        self.lr_mix = lr_mix
        self.lr_under = lr_under
        self.lr_back = lr_back
        self.mixing_model = mixing_model
        self.l1_mult = l1_mult
        self.l2_mult = l2_mult
        self.nonexc_mult = nonexc_mult
        self.cosine_mult = cosine_mult
    
    def compile(self,ims,overtext_ims,im_list):

        tf.reset_default_graph()
        graph = tf.get_default_graph()
        #create Inverse model
        if len(im_list)>1:
            latent_variables_utb = [np.random.randn(self.nb_classes, self.z_utb) for _ in range(len(im_list))]
            latent_variables_b = [np.random.randn(self.nb_classes, self.z_b) for _ in range(len(im_list))]
        else:
            latent_variables_utb = [np.expand_dims(np.random.randn(self.nb_classes, self.z_utb),0)]
            latent_variables_b = [np.expand_dims(np.random.randn(self.nb_classes, self.z_b),0)]

        nb_ch=ims[0].shape[2]
        InvModel=LatentInvModel(z_utb=self.z_utb,z_b=self.z_b,image_size=self.image_size,graph=graph,mask=self.mask,\
                                gen_restore_path_utb=self.gen_restore_path_utb,gen_restore_path_b=self.gen_restore_path_b,\
                                data_dir=self.data_dir,ganloss=self.ganloss,batch_size=len(ims),model_idx=None,\
                                nb_obs=nb_ch,with_background=self.with_background, restore_mix=restore_mix, mixing_model=self.mixing_model,\
                                l1_mult=self.l1_mult, l2_mult=self.l2_mult, nonexc_mult=self.nonexc_mult, cosine_mult=self.cosine_mult)

        optimizer_mix,cost_mix,latent_var_undertext,latent_var_background,org_mix,overtext_mix,undertext_mix,est_mix,summary_mix = InvModel.build_model_mixmodel(self.lr_mix)
        for ims_idx in range(len(im_list)):     
            for init_idx in range(self.nb_classes):
                gpu_idx=0 if ims_idx%2==0 else 0
                InvModel.build_model_1_latent_v(self.lr_under,self.lr_back,ims_idx,latent_variables_utb[ims_idx][init_idx],latent_variables_b[ims_idx][init_idx], init_idx, gpu_idx=gpu_idx,
                                                im=np.expand_dims(ims[ims_idx],0),overtext_im=np.expand_dims(overtext_ims[ims_idx],0))#1.2e-3
        return InvModel,optimizer_mix,cost_mix,latent_var_undertext,latent_var_background,org_mix,overtext_mix,undertext_mix,est_mix,summary_mix

    def initialize_model(self,inv_model):
        inv_model.create_model()

    def train(self,inv_model,nb_samples,optimizer_mix,cost_mix,latent_var_undertext,latent_var_background,org_mix,overtext_mix,undertext_mix,est_mix,summary_mix):
        # number of iterations
        T = self.out_iter
        T1 = self.in_iter
        nb_obs = self.nb_classes
        mix_write = tf.summary.FileWriter(os.path.join(self.log_dir_inverse, 'logs'), inv_model.graph)
        optimize_list_under = []
        optimize_list_back = []
        input_vectors_images = []
        for ims_idx in range(nb_samples):
            for init_idx in range(nb_obs):
                cur_dict_under = inv_model.latent_models_dict[(ims_idx,init_idx)]
                cur_dict_under["optimizer"] = inv_model.optimizers_under[(ims_idx,init_idx)]
                cur_dict_back = inv_model.latent_models_dict[(ims_idx, init_idx)]
                cur_dict_back["optimizer"] = inv_model.optimizers_back[(ims_idx, init_idx)]
                optimize_list_under.append(cur_dict_under)
                optimize_list_back.append(cur_dict_back)
                input_vectors_images.append(inv_model.latent_models_dict[(ims_idx, init_idx)])


        for epoch in range(T):
            start_time = time.time()
            out = inv_model.sess.run(input_vectors_images)
            #prepare latent variables + ims samples
            samples = list(
                zip([np.squeeze(out[idx]["latent_variables_undertext"],0) for idx in range(nb_samples * nb_obs)],
                    [np.squeeze(out[idx]["latent_variables_background"],0) for idx in range(nb_samples * nb_obs)], \
                    [np.squeeze(out[idx]["org_plh"],0) for idx in range(nb_samples * nb_obs)],
                    [np.squeeze(out[idx]["overtext_plh"],0) for idx in range(nb_samples * nb_obs)]))
            latent_v_utb,latent_v_b,samples_ims,overtext_samples = zip(*samples)
            if not self.lr_mix==0:
                for t1 in range(T1):

                    _,gen_loss,undertext_gen,org_gen,est_gen,summ_buf = inv_model.sess.run((optimizer_mix, cost_mix, undertext_mix,org_mix,est_mix,summary_mix),
                                                feed_dict={org_mix: samples_ims, overtext_mix: overtext_samples,\
                                                           latent_var_undertext:latent_v_utb,latent_var_background:latent_v_b})
                
                print("epoch {}: genloss {},  time {}".format(epoch, np.mean(gen_loss),time.time()-start_time))
            else:
                
                summ_buf = inv_model.sess.run(summary_mix,
                                                feed_dict={org_mix: samples_ims, overtext_mix: overtext_samples,\
                                                           latent_var_undertext:latent_v_utb,latent_var_background:latent_v_b})
            mix_write.add_summary(summ_buf,epoch)
            for t1 in range(T1):
                out = inv_model.sess.run(optimize_list_under)
            cost = [x["cost"] for x in out]
            print("epoch {}: latentloss {}, time {}".format(epoch, np.mean(cost), time.time() - start_time))
            if self.with_background:
                for t1 in range(T1):
                    out = inv_model.sess.run(optimize_list_back)
            cost = [x["cost"] for x in out]
            mix_write.add_summary(out[0]["summ"],epoch)
            print("epoch {}: backloss {}, time {}".format(epoch,np.mean(cost), time.time()-start_time))
        mix_write.close()
        inv_model.saver.save(inv_model.sess,os.path.join(self.log_dir_inverse,str(batch_num)),global_step=epoch)

    def save_the_best(self,inv_model,ims,im_list):
        gen_undertext=[]
        gen_overlap=[]
        gen_background = []
        losses_dict={}
        nb_ch = ims[0].shape[2]#self.nb_classes
        nb_samples = len(im_list)
        for ims_idx in range(nb_samples):
            gen_loss = inv_model.sess.run([inv_model.latent_models_dict[(ims_idx,i)]["cost"] for i in range(self.nb_classes)])
            print("gen loss",gen_loss)
            best_idx = np.argmin(np.array(gen_loss))
            losses_dict[im_list[ims_idx]]=str(gen_loss[best_idx])
            if self.with_background:
                im_undertext,im_overlap,im_background = inv_model.sess.run((inv_model.latent_models_dict[(ims_idx,best_idx)]['undertext'],
                                                         inv_model.latent_models_dict[(ims_idx,best_idx)]['est'],\
                                                         inv_model.latent_models_dict[(ims_idx,best_idx)]['background']))
            else:
                im_undertext, im_overlap = inv_model.sess.run(
                    (inv_model.latent_models_dict[(ims_idx, best_idx)]['undertext'],
                     inv_model.latent_models_dict[(ims_idx, best_idx)]['est']))
            print("overlap shape",im_overlap.shape)
            gen_undertext.append(np.squeeze(im_undertext,axis=(0,3)))
            if nb_ch>1:
                gen_overlap.append(np.squeeze(im_overlap, axis=(0)))
            else:
                gen_overlap.append(np.squeeze(im_overlap,axis=(0,3)))
            if with_background:
                gen_background.append(np.squeeze(im_background,axis=(0,3)))
        inv_model.sess.close()
        if self.with_background:
            self.save_predicted_ims(im_list, under_ims=gen_undertext,overlap_ims=gen_overlap,background_ims=gen_background,nb_obs=nb_ch)
        else:
            self.save_predicted_ims(im_list, under_ims=gen_undertext,overlap_ims=gen_overlap)
        with open(os.path.join(self.log_dir,"losses.json"),'w') as fp:
            json.dump(losses_dict, fp)

    def train_batch(self,ims,overtext_ims,im_list):
        inv_model,optimizer_mix,cost_mix,latent_var_undertext,latent_var_background,org_mix,overtext_mix,undertext_mix,est_mix,summary_mix = self.compile(ims,overtext_ims,im_list)
        self.initialize_model(inv_model)
        nb_samples = len(im_list)
        #clean summary
        summary_dir =  os.path.join(self.log_dir_inverse, 'logs')
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)

        self.train(inv_model,nb_samples,optimizer_mix,cost_mix,latent_var_undertext,latent_var_background,org_mix,overtext_mix,undertext_mix,est_mix,summary_mix)
        self.save_the_best(inv_model,ims,im_list)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Extract features from layers using provided images")
    parser.add_argument("-m", "--mask", type=float, default=0.0, help="mask level, if zero no attention map")
    parser.add_argument("-exp", "--exp_name", type=str, default="TEST", help="Experiment name prefix")
    parser.add_argument("-bs","--batch_size", type=int, help="model batch_size")
    parser.add_argument("-ld", "--log_dir", type=str,  help="Log dir, root directory for model folders")
    parser.add_argument("-rp_under", "--restore_path_under", type=str,  help="Restore path for undertext, restore generator model path")
    parser.add_argument("-rp_back", "--restore_path_back", type=str, default="", help="Restore path for background, restore generator model path.If empty then do not use background")
    parser.add_argument("-dd", "--data_dir", type=str,  help="Data dir, path to directory with images")
    parser.add_argument("-ims", "--im_size", type=int, help="Image size")
    parser.add_argument("-zutb", "--z_utb", type=int, help="Dimension of latent vectors normal part for undertext")
    parser.add_argument("-zb", "--z_b", type=int, help="Dimension of latent vectors normal part for background")
    parser.add_argument("-K", "--nb_classes", type=int, help="Dimension of latent vectors one hot vector part or it can be be a number of initializations if latent vector does not have categorical part")
    parser.add_argument("-gloss", "--ganloss", type=float,default=0, help="Not zero - add Generator loss")
    parser.add_argument("-it", "--initer", type=int,default=1, help="Number of iteration in inner loop")
    parser.add_argument("-ot", "--outiter", type=int,default=300, help="Number of iteration in outer loop")
    parser.add_argument("-nob", "--nobs", type=int,default=1, help="Number of observations")
    parser.add_argument("-l", "--list", type=str, nargs="+",default=[], help="list of files")
    parser.add_argument("-remix", "--restore_mix", type=str, help="Whether to restore mixing network")
    parser.add_argument("-lrm", "--lr_mix", type=float, default=0, help="Mixing Network loss")
    parser.add_argument("-lru", "--lr_un", type=float, default=1.2e-3, help="Undertext loss")
    parser.add_argument("-lrb", "--lr_back", type=float, default=1.2e-3, help="Background loss")
    parser.add_argument("-mnet", "--mix_net", type=str, help="Mixing model")
    parser.add_argument("-l1", "--l1mult", type=float, default=1.0, help="Multiplier for l1 loss")
    parser.add_argument("-l2", "--l2mult", type=float, default=0.0, help="multiplier for l2 loss")
    parser.add_argument("-lexc", "--lexc", type=float, default=0.0, help="multiplier for exclusion loss")
    parser.add_argument("-lcos", "--lcos", type=float, default=0.0, help="multiplier for cosine similarity loss")
    FLAGS, unparsed = parser.parse_known_args()
    # change current dirrectory, ../
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logdir = FLAGS.log_dir
    gen_restore_path_under = FLAGS.restore_path_under
    gen_restore_path_back = FLAGS.restore_path_back
    data_dir = FLAGS.data_dir
    image_size = FLAGS.im_size
    batch_size = FLAGS.batch_size
    model_name = FLAGS.exp_name.lower()
    restore_mix = json.loads(FLAGS.restore_mix.lower())
    if FLAGS.lr_mix==0:
        assert  restore_mix==True, "You could not set learning rate of mixing network to 0 if you are not using pretrained network"
    seed=1111
    random.seed(seed)
    np.random.seed(seed)
    if not FLAGS.list:
        im_list = make_im_list(data_dir)
        random.shuffle(im_list)
    else:
        im_list = [os.path.join(data_dir,fname.rstrip()) for fname in FLAGS.list]
    exp_name = FLAGS.exp_name
    ganloss = FLAGS.ganloss
    if len(FLAGS.restore_path_back)!=0:
        with_background = True
        exp_name = exp_name + "_wback"
    else:
        with_background = False
        exp_name = exp_name + "_noback"
    if restore_mix:
        exp_name = exp_name + "_rmix"
    if FLAGS.lr_mix>0:
        exp_name = exp_name + "tune"
    if FLAGS.mask>0:
        exp_name = exp_name+'_mask{}'.format(FLAGS.mask)
    if FLAGS.nb_classes>0:
        exp_name = exp_name+'_init{}'.format(FLAGS.nb_classes)

    exp_name = exp_name+'_it{}'.format(FLAGS.initer)
    exp_name = exp_name+'_ot{}'.format(FLAGS.outiter)
    exp_name = exp_name+'_obs{}'.format(FLAGS.nobs)
    if 'archi' in data_dir.lower():
        _set=os.path.split(data_dir)[1]
    else:
        print("There is no recognized data name for {}".format(data_dir))
    print("Set",_set)
    #create an object that trains Reconstruction network
    timestep = strftime("%Y-%m-%d-%H-%M", gmtime())
    logs = BSSModelTraining(log_dir=os.path.join(logdir,exp_name),gen_restore_path_utb=gen_restore_path_under,\
                             gen_restore_path_b=gen_restore_path_back,image_size=image_size,dataset=_set,\
                             mask=FLAGS.mask,nb_utb=FLAGS.z_utb,nb_b=FLAGS.z_b,\
                             nb_classes=FLAGS.nb_classes, model_name=model_name,\
                             data_dir=data_dir,ganloss=ganloss,in_iter=FLAGS.initer,\
                             out_iter=FLAGS.outiter, with_background=with_background, restore_mix=restore_mix, lr_mix = FLAGS.lr_mix,
                             lr_under = FLAGS.lr_un, lr_back = FLAGS.lr_back, mixing_model = FLAGS.mix_net,timestep=timestep,
                             l1_mult=FLAGS.l1mult,l2_mult=FLAGS.l2mult,nonexc_mult=FLAGS.lexc,cosine_mult=FLAGS.lcos)
    logs.prepare_log_dirs(im_list,nb_obs=FLAGS.nobs)
    #save experiment parameters in a dictionary
    dict_to_json(os.path.join(FLAGS.log_dir,exp_name,timestep,"model_param.json"), FLAGS.__dict__)
    FLAGS.log_dir_inverse = logs.log_dir_inverse
    if not FLAGS.list:
        batch_num=0
        for idx in range(0,len(im_list),batch_size):
            batch, overtext_batch = read_batch(im_list[idx:min(idx+batch_size,len(im_list))],image_size,data_dir=data_dir, nb_obs=FLAGS.nobs)
            logs.train_batch(batch,overtext_batch,im_list[idx:min(idx+batch_size,len(im_list))])
            batch_num = batch_num+1
    else:
        batch, overtext_batch = read_batch(im_list,image_size,data_dir=data_dir,nb_obs=FLAGS.nobs)
        logs.train_batch(batch,overtext_batch,im_list)

