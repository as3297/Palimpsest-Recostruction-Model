from image import *
import tensorflow as tf
from random import shuffle, choice
from skimage import io, transform
from matplotlib import pyplot as plt
from tensorflow.python.ops import variable_scope,init_ops
from math import sqrt
from time import gmtime, strftime
from undertext_gen_train import Text_Back_Gen
import time
import json


class Text_Back_Finetune(Text_Back_Gen):
    def __init__(self, batch_size, with_overtext,mix_func,mode,restore_path_back,timestep, restore_path_undertext = None, train_background = False):
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
        self.train_background = train_background
        self.timestep = timestep

        self.log_dir = "/home/as3297/projects/bss_gan_palimpsest/training/DCGAN/finetune_greek_mode_full_data_{}" \
                      "{}zb_{}gut_{}dut_{}".format(
                self.mode, self.dim_z_b, self.dim_g_b, self.dim_d_ut, self.timestep)



    def compile_model_uto(self,z_b=None):
        self.has_otb = tf.placeholder(tf.bool, shape=(), name="overtext_indicator")
        self.ut_real = tf.placeholder(tf.float32, shape=(None, self.image_size,self.image_size,1),name="ut_real")
        self.utb_real = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1), name="utb_real")
        back_fake = self.gen_background(z_b)
        if self.with_overtext:
            self.ot_real = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1), name="overtext")
            ot_real_rand = tf.cond(self.has_otb, lambda: self.ot_real, lambda: tf.zeros_like(self.utb_real) - 1.0)
            self.summary_background.append(tf.summary.image("overtext_real", ot_real_rand))
            ut_fake = self.mixing_net(self.utb_real,back_fake,ot_real_rand)
            self.under_overtext = tf.multiply(1 - (self.utb_real + 1) / 2, 1 - (ot_real_rand + 1) / 2)
            ut_no_text = self.mixing_net(tf.zeros_like(self.utb_real) - 1.0, back_fake,
                                         tf.zeros_like(self.utb_real) - 1.0)
            ut_no_overtext = self.mixing_net(self.utb_real, back_fake,
                                               tf.zeros_like(self.utb_real) - 1.0)
            self.disc_antiovertext_fake = self.disc_ut(ut_no_overtext)
            self.uto_no_background = self.mixing_net(self.utb_real,tf.zeros_like(self.utb_real)-1, ot_real_rand)
        else:
            ut_fake = self.mixing_net(self.utb_real, back_fake)
            self.uto_no_background = self.mixing_net(self.utb_real, tf.zeros_like(self.utb_real)-1)
            self.under_overtext = 1 - (self.utb_real + 1) / 2
            ut_no_text = self.mixing_net(tf.zeros_like(self.utb_real) - 1.0, back_fake)


        self.summary_background = self.summary_background + [tf.summary.image("without_back",self.uto_no_background),
                                       tf.summary.image("after_mixing", ut_fake),
                                       tf.summary.image("ut_real", self.ut_real),
                                       tf.summary.image("background", back_fake),
                                       tf.summary.image("under_overtext", tf.squeeze(tf.slice(tf.image.sobel_edges(self.under_overtext),[0,0,0,0,0],[3,self.image_size,self.image_size,1,1]),3)),
                                       tf.summary.image("no_background", tf.squeeze(tf.slice(tf.image.sobel_edges(self.uto_no_background),[0,0,0,0,0],[3,self.image_size,self.image_size,1,1]),3))]
        self.disc_ut_fake = self.disc_ut(ut_fake)
        self.disc_ut_real = self.disc_ut(self.ut_real)
        self.disc_antitext_fake = self.disc_ut(ut_no_text)


    def train_gen_ut(self):
        loss_gen_ut = self.loss_dcgan_gen(self.disc_ut_fake)
        antitext_loss = self.loss_anti_shortcut(self.disc_antitext_fake)
        antiback_loss = tf.losses.sigmoid_cross_entropy(tf.image.sobel_edges(self.uto_no_background),tf.image.sobel_edges(self.under_overtext))
        loss = loss_gen_ut + 100.0*antitext_loss+antiback_loss
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
        if overtext:
            imdirs = ["/home/as3297/projects/bss_gan_palimpsest/datasets/Archimedes/test_obscured_thresholded"]
        else:
            imdir1 = "/home/as3297/projects/bss_gan_palimpsest/datasets/Archimedes/test_obscured"
            imdir2 = "/home/as3297/projects/bss_gan_palimpsest/datasets/Archimedes/train_clean_no_PsiXiBetaZeta"
            imdir3 = "/home/as3297/projects/bss_gan_palimpsest/datasets/Archimedes/test_clean_no_PsiXiBetaZeta"
            imdirs = [imdir1,imdir2,imdir3]
        #org_im_list = os.listdir(shortcutdir)
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

    def finetune(self,attr_dict,init_epoch=0):
        self.compile_model_uto()
        disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf = self.train_disc_ut()
        gen_ut_train_op, gen_ut_loss, gen_ut_summ = self.train_gen_ut()
        z_test = np.random.normal(0, 1, [32, self.dim_z_b])
        summ_back_gen = self.test_background(z_test,self.log_dir,init_epoch,self.mixing_net,overtext=self.with_overtext)

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
                             self.has_otb: np.random.choice([True,False],p=[0.3,0.7])}

                for _ in range(5):
                    _, disc_loss_ut, disc_ut_summ = sess.run([disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf],
                                                             feed_dict=feed_dict)  # bool(np.random.choice([0,1],p=[0.3,0.7]))})
                print("Epoch:{}, disc_ut loss:{}, time:{:.3f}".format(iteration, disc_loss_ut,
                                                                          time.time() - start_time))
                writer.add_summary(disc_ut_summ, iteration)


                feed_dict = {
                             self.utb_real: self.sample_ut(self.batch_size,binary=True),
                             self.ot_real: self.sample_uto(self.batch_size,overtext=True),
                             self.has_otb: np.random.choice([True,False],p=[0.3,0.7])}
                _, gen_loss, gen_back_summ = sess.run([gen_ut_train_op, gen_ut_loss, gen_ut_summ],
                                                      feed_dict=feed_dict)
                writer.add_summary(gen_back_summ, iteration)
                if iteration % 100 == 0:
                    gen_back_summ = sess.run(summ_back_gen)
                    writer.add_summary(gen_back_summ, iteration)
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=iteration)
        writer.close()

if __name__=="__main__":
    main_dir = "/home/as3297/projects/bss_gan_palimpsest/training/DCGAN/"
    init_epoch=0
    back_dir = "greek_mode_ut_background_antishort_36lutb_16lb_32gutb_8gut_16dut_128dutb/model-19999"#"finetune_greek_mode_full_data_ut16zb_8gut_16dut_2020-03-10-20-51/model-{}".format(init_epoch)#
    under_dir = None#os.path.join(main_dir,"greek_mode_utb_background_antishort_30lutb_16lb_32gutb_8gut_16dut_8dutb/model-19999")
    timestep = strftime("%Y-%m-%d-%H-%M", gmtime())
    mixing_func = "mixing_net_1d"
    model = Text_Back_Finetune(batch_size=64, with_overtext=True,mix_func=mixing_func,mode='ut',\
                               restore_path_back=os.path.join(main_dir,back_dir),restore_path_undertext=under_dir,train_background=False,timestep=timestep)
    attr_dict = vars(model)
    attr_dict_str = {}
    for key,val in attr_dict.items():
        attr_dict_str[key]=str(val)
    log_dir = model.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "parameters.json"),
              'a') as fp:
        json.dump(attr_dict_str,fp)

    model.finetune(attr_dict,init_epoch)
