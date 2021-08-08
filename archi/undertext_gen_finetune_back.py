import os
import numpy as np
import tensorflow as tf
from random import shuffle
from skimage import io, transform
from time import gmtime, strftime
from undertext_gen_train import Text_Back_Gen
from utils import save_images_gan
import datetime
import time
import json


class Text_Back_Finetune(Text_Back_Gen):
    def __init__(self, batch_size, with_overtext,mix_func,mode,restore_path_back,timestep, non_exc_loss, restore_path_undertext = None):
        super().__init__(batch_size)
        self.dim_d_ut = 16
        self.with_overtext = with_overtext
        self.disc_ut_lr = 1e-5
        self.nb_epochs = 20000
        self.mix_ker = 3
        self.gen_ut_lr = 1e-5
        self.mix_lr = 1e-4
        self.mixing_net = eval("self."+mix_func.lower())
        self.mode = mode
        self.restore_path_back = restore_path_back
        self.restore_path_undertext = restore_path_undertext
        self.timestep = timestep
        self.non_exc_loss_scalar = non_exc_loss

        self.log_dir = "C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/test/finetune_greek_mode_full_data_{}" \
                      "{}zb_{}gut_{}dut_excloss{}_{}_{}".format(
                self.mode, self.dim_z_b, self.dim_g_b, self.dim_d_ut,str(self.non_exc_loss_scalar).replace(".",""),mix_func.lower(), self.timestep)

    def test_background(self, log_dir, init_epoch, save=None, overtext=False, z_b=[]):

        if len(z_b)!=0:
            nb_samples = len(z_b)
            z_b = tf.constant(z_b, dtype=tf.float32)
            back_fake = self.gen_background(z_b)
        else:
            back_fake = self.gen_background()
            nb_samples = self.batch_size
        if not overtext:
            back_after_mix = self.mixing_net(
                tf.zeros(dtype=tf.float32, shape=(nb_samples, self.image_size, self.image_size, 1)) - 1.0, back_fake)

        else:
            back_after_mix = self.mixing_net(
                tf.zeros(dtype=tf.float32, shape=(nb_samples, self.image_size, self.image_size, 1)) - 1.0,
                back_fake, tf.zeros(dtype=tf.float32, shape=(nb_samples, self.image_size, self.image_size, 1)) - 1.0)
        restore_var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack/Gen") + \
                           tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="MixingNet")
        if save!=None:
            self.compile_model_uto()
            if not os.path.isdir(os.path.join(log_dir, "test_background")):
                os.makedirs(os.path.join(log_dir, "test_background"))
            saver = tf.train.Saver(restore_var_list)
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(log_dir, 'model-' + str(init_epoch)))
                background_pred = sess.run(back_after_mix)
                feed_dict = {self.utb_real: self.sample_ut(nb_samples, binary=True),
                             self.ot_real: self.sample_uto(nb_samples, overtext=True),
                             self.has_otb: np.array(self.batch_size*[True])}
                palimp_pred= sess.run(self.ut_fake, feed_dict = feed_dict)
                background_pred = ((0.5*background_pred+0.5)*255).astype(np.uint8)
                palimp_pred = ((0.5*palimp_pred+0.5)*255).astype(np.uint8)
                print("palimp min {}, palimp max {}".format(np.min(palimp_pred), np.max(palimp_pred)))
                now = datetime.datetime.now()
                save_images_gan(np.squeeze(background_pred, axis=3),
                                os.path.join(log_dir, "test_background", "back" + "_" + str(init_epoch) + ".png"))
                save_images_gan(np.squeeze(palimp_pred, axis=3),
                                os.path.join(log_dir, "test_background", "pred" + "_" + str(init_epoch) + ".png"))
        else:
                return tf.summary.image("after_mixing_without_text", back_after_mix, max_outputs=self.batch_size)

    def compile_model_uto(self,z_b=None):
        self.has_otb = tf.placeholder(tf.bool, shape=self.batch_size, name="overtext_indicator")
        self.ut_real = tf.placeholder(tf.float32, shape=(None, self.image_size,self.image_size,1),name="ut_real")
        self.utb_real = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1), name="utb_real")
        back_fake = self.gen_background(z_b)
        if self.with_overtext:
            self.ot_real = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1), name="overtext")
            self.ot_real_rand = tf.where(self.has_otb,self.ot_real,-1*tf.ones_like(self.utb_real) )
            self.summary_background.append(tf.summary.image("overtext_real", self.ot_real_rand))
            self.ut_fake = self.mixing_net(self.utb_real,back_fake,self.ot_real_rand)
            ut_no_text = self.mixing_net(tf.zeros_like(self.utb_real) - 1.0, back_fake,
                                         tf.zeros_like(self.utb_real) - 1.0)
            ut_no_overtext = self.mixing_net(self.utb_real, back_fake,
                                               tf.zeros_like(self.utb_real) - 1.0)
            self.disc_antiovertext_fake = self.disc_ut(ut_no_overtext)
            self.utb_no_background = self.mixing_net(self.utb_real,-1*tf.ones_like(self.utb_real), -1*tf.ones_like(self.utb_real))
            self.ot_no_background = self.mixing_net( -1*tf.ones_like(self.utb_real), -1*tf.ones_like(self.utb_real), self.ot_real_rand)

        else:
            self.ut_fake = self.mixing_net(self.utb_real, back_fake)
            self.under_overtext = 1 - (self.utb_real + 1) / 2
            ut_no_text = self.mixing_net(tf.zeros_like(self.utb_real) - 1.0, back_fake)
            self.utb_no_background = self.mixing_net(self.utb_real, -1 * tf.ones_like(self.utb_real))


        self.summary_background = self.summary_background + [tf.summary.image("without_back_utb",self.utb_no_background),
                                       tf.summary.image("after_mixing", self.ut_fake),
                                       tf.summary.image("ut_real", self.ut_real),
                                       tf.summary.image("background", back_fake),
                                       tf.summary.image("under_edge", tf.expand_dims(self.gradient_mag(self.utb_real),axis=3)),
                                       tf.summary.image("under_mix_edge", tf.expand_dims(self.gradient_mag(self.utb_no_background),axis=3))]
        if self.with_overtext:
            self.summary_background += [tf.summary.image("without_back_ot",self.ot_no_background),
                                        tf.summary.image("over_edge", tf.expand_dims(self.gradient_mag(self.ot_real_rand),axis=3)),
                                       tf.summary.image("over_mix_edge", tf.expand_dims(self.gradient_mag(self.ot_no_background),axis=3))]
        self.disc_ut_fake = self.disc_ut(self.ut_fake)
        self.disc_ut_real = self.disc_ut(self.ut_real)
        self.disc_antitext_fake = self.disc_ut(ut_no_text)

    def gradient_mag(self,x,nb_ch=1):
        sobel_x = tf.image.sobel_edges(x)
        grad_x = tf.slice(sobel_x, [0, 0, 0, 0, 0], [tf.shape(x)[0], self.image_size, self.image_size, nb_ch, 1])
        grad_y = tf.slice(sobel_x, [0, 0, 0, 0, 1], [tf.shape(x)[0], self.image_size, self.image_size, nb_ch, 1])
        eps = 1e-9
        sum_sq = grad_x ** 2 + grad_y ** 2
        mag = tf.reduce_mean(tf.sqrt(tf.maximum(sum_sq,eps)),axis=[3,4])
        return tf.divide(mag, tf.reshape(tf.reduce_max(mag,axis=[1,2]),[tf.shape(x)[0], 1, 1]))

    def non_exc_loss(self,bare_text,text):
        """calculate the loss between undertext and undertext propagated through mixing model in the absence of overtext and background.
        Suppose to prevent mixing model to suppress undertext and substitute with paterns from background
        """
        mag_bare_text = self.gradient_mag(bare_text)
        mag_text = self.gradient_mag(text)
        non_exc_loss = self.l1_loss(tf.expand_dims(mag_bare_text,axis=3),tf.expand_dims(mag_text, axis=3))#
        return non_exc_loss

    def l1_loss(self, pred, org, mask=None, alpha=0.0):
        """
        mask - overtext mask, shape [batch,channel,width,height]
        alpha - value of overtext
        """
        if mask != None:
            mask = tf.clip_by_value(t=mask, clip_value_min=alpha, clip_value_max=1, )
            recon_loss = tf.reduce_mean(tf.multiply(mask, tf.abs(pred - org)))

        else:
            recon_loss = tf.reduce_mean(tf.abs(pred - org),axis=[1,2,3])
        return recon_loss

    def train_gen_ut(self):
        loss_gen_ut = self.loss_dcgan_gen(self.disc_ut_fake)
        antitext_loss = 100.0*self.loss_anti_shortcut(self.disc_antitext_fake)
        if self.with_overtext:
            antiback_loss = self.non_exc_loss_scalar*tf.where(self.has_otb, tf.reduce_mean([self.non_exc_loss(self.ot_no_background,(self.ot_real_rand + 1) / 2),
                                                                 self.non_exc_loss(self.utb_no_background, (self.utb_real + 1) / 2)],axis=0),
                                                              self.non_exc_loss(self.utb_no_background,(self.utb_real + 1) / 2))
        else:
            antiback_loss = self.non_exc_loss_scalar * self.non_exc_loss(self.utb_no_background,(self.utb_real + 1) / 2)
        antiback_loss = tf.reduce_mean(antiback_loss)
        loss = loss_gen_ut +antitext_loss+antiback_loss
        train_op_1 = tf.train.AdamOptimizer(self.gen_ut_lr).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="TextBack/Gen"))

        train_op_2 = tf.train.AdamOptimizer(self.mix_lr).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="MixingNet"))
        train_op = tf.group(train_op_1,train_op_2)
        summ_loss_ut = tf.summary.scalar("loss_gen_ut",loss)
        summ_loss_antitext = tf.summary.scalar("loss_gen_notext",antitext_loss)
        summ_loss_antiback = tf.summary.scalar("loss_antiignore_text",antiback_loss)
        summary = tf.summary.merge([summ_loss_ut,summ_loss_antitext,summ_loss_antiback])
        return train_op,loss_gen_ut,summary

    def train_disc_ut(self):
        loss_disc_ut = self.loss_dcgan_disc(self.disc_ut_real,self.disc_ut_fake)
        train_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack/Disc/Discriminator")
        train_vars = train_vars1
        train_op = tf.train.AdamOptimizer(self.disc_ut_lr).minimize(loss_disc_ut, var_list = train_vars)
        summ_ut = tf.summary.scalar("loss_disc_ut", loss_disc_ut)
        summary = tf.summary.merge([summ_ut]+self.summary_background)
        return train_op, loss_disc_ut, summary

    def sample_uto(self,batch_size,overtext):
        """Depending on the overtext flag returns either grayscale images of palimpsest, or binary images of overtext """
        data_dir_par = r"C:/Data/PhD/bss_gan_palimpsest"
        if overtext:
            imdirs = [data_dir_par+"/datasets/Archimedes/test_obscured_thresholded"]
        else:
            imdir1 = data_dir_par+"/datasets/Archimedes/test_obscured"
            imdir2 = data_dir_par+"/datasets/Archimedes/train_clean_no_PsiXiBetaZeta"
            imdir3 = data_dir_par+"/datasets/Archimedes/test_clean_no_PsiXiBetaZeta"
            imdirs = [imdir1,imdir2,imdir3]
        org_im_list = []
        for imdir in imdirs:
            for char in os.listdir(imdir):
                for fname in os.listdir(os.path.join(imdir,char)):
                    org_im_list.append(os.path.join(imdir,char,fname))
        idx_list = list(range(len(org_im_list)))
        shuffle(idx_list)
        im_paths = [org_im_list[i] for i in idx_list[0:batch_size]]
        ims = [io.imread(fname=im_path, as_gray=True) for im_path in im_paths]
        ims = [transform.resize(im, (self.image_size, self.image_size)) for im in ims]
        ims = np.array(ims).reshape(-1, self.image_size, self.image_size, 1)
        if np.max(ims) > 1:
            print("Scale image  contrast")
            ims = ims/np.max(ims)

        if overtext:
            ims = 1-ims
            ims = 2.0 * ims - 1.0
        else:
            ims = 2.0 * ims/np.max(ims) - 1.0
            for im_idx in range(len(ims)):
                contrast = np.random.uniform(0.2,1)
                ims[im_idx]= (ims[im_idx])*contrast
        return ims

    def finetune(self,init_epoch=0):
        self.compile_model_uto()
        disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf = self.train_disc_ut()
        gen_ut_train_op, gen_ut_loss, gen_ut_summ = self.train_gen_ut()
        z_test = np.random.normal(0, 1, [32, self.dim_z_b])
        summ_back_gen = self.test_background(log_dir=self.log_dir,init_epoch=init_epoch,save=None,overtext=self.with_overtext,z_b=z_test)

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
        print("Working directory changed to: ", self.log_dir)
        writer = tf.summary.FileWriter(self.log_dir + '/logs', tf.get_default_graph())
        # Train loop
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            print("Initialized values")
            if self.restore_path_back!=None:
                if self.mode=="ut":
                    restore_var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack")
                    saver_restore = tf.train.Saver(restore_var_list)
                    saver_restore.restore(sess, os.path.join(self.restore_path_back))
                    print("Restore background generator")
                    if init_epoch > 0:
                        restore_var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="MixingNet")
                        saver_restore = tf.train.Saver(restore_var_list)
                        saver_restore.restore(sess, os.path.join(self.restore_path_back))
                        print("Restore Mixing Network")

            for iteration in range(init_epoch+1, self.nb_epochs):
                start_time = time.time()

                feed_dict = {self.ut_real: self.sample_uto(self.batch_size,overtext=False),
                             self.utb_real: self.sample_ut(self.batch_size,
                             binary=True),self.ot_real: self.sample_uto(self.batch_size,overtext=True),
                             self.has_otb: np.random.choice([True,False],p=[0.3,0.7],size=self.batch_size,replace=True)}

                for _ in range(5):
                    _, disc_loss_ut, disc_ut_summ = sess.run([disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf],
                                                             feed_dict=feed_dict)  # bool(np.random.choice([0,1],p=[0.3,0.7]))})
                print("Epoch:{}, disc_ut loss:{}, time:{:.3f}".format(iteration, disc_loss_ut,
                                                                          time.time() - start_time))
                writer.add_summary(disc_ut_summ, iteration)


                feed_dict = {
                             self.utb_real: self.sample_ut(self.batch_size,binary=True),
                             self.ot_real: self.sample_uto(self.batch_size,overtext=True),
                             self.has_otb: np.random.choice([True,False],p=[0.3,0.7],size=self.batch_size)}
                _, gen_loss, gen_back_summ = sess.run([gen_ut_train_op, gen_ut_loss, gen_ut_summ],
                                                      feed_dict=feed_dict)
                writer.add_summary(gen_back_summ, iteration)
                if iteration % 100 == 0:
                    gen_back_summ = sess.run(summ_back_gen)
                    writer.add_summary(gen_back_summ, iteration)
                    saver.save(sess, os.path.join(self.log_dir, 'model'), global_step=iteration)
        writer.close()

def run_model():
    main_dir = r"C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/"
    init_epoch = 0
    back_dir = r"greek_mode_ut_background_antishort_36lutb_16lb_32gutb_8gut_16dut_128dutb/model-19999"
    under_dir = None
    timestep = strftime("%Y-%m-%d-%H-%M", gmtime())
    mixing_func = "mixing_net_2d"
    model = Text_Back_Finetune(batch_size=16, with_overtext=True, mix_func=mixing_func, mode='ut', \
                               restore_path_back=os.path.join(main_dir, back_dir), restore_path_undertext=under_dir,
                               train_background=False, timestep=timestep, non_exc_loss=0.1)
    save_parameters(model)
    model.finetune(init_epoch)

def save_parameters(model):
    attr_dict = vars(model)
    attr_dict_str = {}
    for key, val in attr_dict.items():
        attr_dict_str[key] = str(val)
    log_dir = model.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "parameters.json"),
              'a') as fp:
        json.dump(attr_dict_str, fp)



if __name__=="__main__":
    main_dir = r"C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/"
    init_epoch=0
    back_dir = r"greek_mode_ut_background_antishort_36lutb_16lb_32gutb_8gut_16dut_128dutb/model-19999"
    under_dir = None
    timestep = strftime("%Y-%m-%d-%H-%M", gmtime())
    mixing_func = "mixing_net_2d"
    model = Text_Back_Finetune(batch_size=16, with_overtext=True,mix_func=mixing_func,mode='ut',\
                               restore_path_back=os.path.join(main_dir,back_dir),restore_path_undertext=under_dir,timestep=timestep, non_exc_loss=0.0)
    save_parameters(model)
    model.finetune(init_epoch)

    for epoch in [19999]:
        try:
            model.test_background(r"C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/greek_mode_ut_background_antishort_36lutb_16lb_32gutb_8gut_16dut_128dutb",epoch,True,True)
        except:
            continue
