import tensorflow as tf
from random import shuffle, choice
from skimage import io, transform
from matplotlib import pyplot as plt
from tensorflow.python.ops import variable_scope,init_ops
from math import sqrt
from utils import save_images_gan
import time
import datetime
import numpy as np
import os


class Text_Back_Gen():
    def __init__(self,batch_size,graph=None,main_dir=None,mode=None,datadir_with_bg=None,datadir_without_bg=None,mix_func=None):

        self.batch_size = batch_size
        self.dim_d_utb = 32#64
        self.dim_d_ut = 32
        self.dim_g_b = 8
        self.dim_g_utb = 32
        self.dim_z_utb = 30
        self.dim_z_b = 16
        self.nb_epochs = 10000
        self.summary_background = []
        self.summary_undertext = []
        self.image_size = 64
        self.mix_ker = 3
        self.disc_ut_lr = 4e-5
        self.disc_utb_lr = 4e-4
        self.gen_utb_lr = 1e-4
        self.gen_ut_lr = 1e-5
        self.mix_lr = 1e-4
        self.main_dir = main_dir
        self.datadir_without_bg = datadir_without_bg
        self.datadir_with_bg = datadir_with_bg
        if not mix_func==None:
            self.mixing_net = eval("self." + mix_func.lower())
        self.mode = mode #"utb" - undertext binary, trains undertext models; "ut" - undertext with back ground, trains background model
        if graph==None:
            graph = tf.get_default_graph()
        with graph.as_default() as graph_def:
            self.utb_real = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1),
                                           name="binary_undertext")
            global_step_scope = "{}/global_step".format("utb")
            self.global_step = variable_scope.get_variable(global_step_scope, [], trainable=False, dtype=tf.int64,
                                                  initializer=init_ops.constant_initializer(0, dtype=tf.int64))


    def test_undertext(self,z_ut,log_dir,init_epoch,save=None,return_im=False):
        nb_samples = len(z_ut)
        z_ut = tf.constant(z_ut,dtype=tf.float32)
        gen_samples = self.gen_underText(z_ut)
        restore_var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBinary/Gen")
        if save:
            if not os.path.isdir(os.path.join(log_dir, "test")):
                os.makedirs(os.path.join(log_dir, "test"))
            saver = tf.train.Saver(restore_var_list)
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(log_dir, 'model-' + str(init_epoch)))
                undertext_pred = sess.run(gen_samples)
                save_images_gan(np.squeeze(undertext_pred,axis=3),os.path.join(log_dir, "test","test"+"_"+str(init_epoch)+".png"))
        if return_im:
            return gen_samples
        else:
            return tf.summary.image("binary_text_gen",gen_samples,max_outputs=nb_samples)

    def test_background(self,z_b,log_dir,init_epoch,mixing_net,save=None,overtext=False):
        nb_samples = len(z_b)
        z_b = tf.constant(z_b,dtype=tf.float32)
        back_fake = self.gen_background(z_b)
        if not overtext:
            back_after_mix = mixing_net(tf.zeros(dtype=tf.float32,shape=(nb_samples, self.image_size,self.image_size,1))-1.0, back_fake)
        else:
            back_after_mix = mixing_net(tf.zeros(dtype=tf.float32,shape=(nb_samples, self.image_size,self.image_size,1)) - 1.0\
                                             , back_fake,tf.zeros(dtype=tf.float32,shape=(nb_samples, self.image_size,self.image_size,1)) - 1.0)
        restore_var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack/Gen")+\
                            tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="MixingNet")
        if save:
            if not os.path.isdir(os.path.join(log_dir, "test_background")):
                os.makedirs(os.path.join(log_dir, "test_background"))
            saver = tf.train.Saver(restore_var_list)
            with tf.Session() as sess:
                saver.restore(sess, os.path.join(log_dir, 'model-' + str(init_epoch)))
                background_pred = sess.run(back_after_mix)
                print("background min {}, background max {}".format(np.min(background_pred),np.max(background_pred)))
                background_pred = background_pred-np.min(background_pred)
                background_pred = background_pred/np.max(background_pred)
                print("background min {}, background max {}".format(np.min(background_pred), np.max(background_pred)))
                now = datetime.datetime.now()
                save_images_gan(np.squeeze(background_pred, axis=3),
                                os.path.join(log_dir, "test_background", "test"+str(now) + "_" + str(init_epoch) + ".png"))
        else:
            return tf.summary.image("after_mixing_without_text",back_after_mix,max_outputs=nb_samples)

    def compile_model_utb(self,z_ut=None):
        utb_fake = self.gen_underText(z_ut)
        self.summary_undertext.append(tf.summary.image("binary_text_real", self.utb_real))
        self.summary_undertext.append(tf.summary.image("binary_text_fake", utb_fake))
        self.disc_utb_fake = self.disc_utb(utb_fake)
        self.disc_utb_real = self.disc_utb(self.utb_real)

    def compile_model_ut(self,z_b=None):
        self.ut_real = tf.placeholder(tf.float32, shape=(None, self.image_size,self.image_size,1),name="undertext")
        self.summary_background.append(tf.summary.image("undertex_real", self.ut_real))
        back_fake = self.gen_background(z_b)
        self.summary_background.append(tf.summary.image("background", back_fake))
        ut_fake = self.mixing_net(self.utb_real,back_fake)
        ut_no_text = self.mixing_net(tf.zeros_like(self.utb_real)-1.0 , back_fake)
        self.summary_background.append(tf.summary.image("after_mixing", ut_fake))
        self.disc_ut_fake = self.disc_ut(ut_fake)
        self.disc_ut_real = self.disc_ut(self.ut_real)
        self.disc_antitext_fake = self.disc_ut(ut_no_text)

    def mixing_net_2d(self,undertext,background=None,overtext=None,nb_obs=1):
        with tf.variable_scope("MixingNet", reuse=tf.AUTO_REUSE):
            if overtext is None:
                source_images = tf.concat([undertext, background], 3)
            if background is None:
                source_images = tf.concat([undertext, overtext], 3)
            if overtext!=None and background!=None:
                source_images = tf.concat([undertext, background,overtext], 3)
            inv1 = tf.layers.conv2d(source_images, 6, self.mix_ker,padding='same',name="mix_h1",
                                    activation='relu',kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
            inv2 = tf.layers.conv2d(inv1, nb_obs, self.mix_ker,padding='same',name="mix_h2",
                                    activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
            return inv2

    def mixing_net_1d(self,undertext,background,overtext=None,nb_obs=1,reshape=True):
        with tf.variable_scope("MixingNet", reuse=tf.AUTO_REUSE):
            if overtext is None:
                source_images = tf.concat([undertext, background], 3)
            else:
                source_images = tf.concat([undertext, background,overtext], 3)

            source_images = tf.reshape(source_images, shape=[-1,3,1])

            inv1 = tf.layers.conv1d(source_images,36, self.mix_ker,padding='same',name="mix_h1",activation='relu',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
            inv2 = tf.layers.conv1d(inv1, nb_obs, self.mix_ker,padding='valid',name="mix_h2",activation=None,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
            if reshape:
                inv2 = tf.reshape(inv2, shape=[-1, self.image_size,self.image_size, nb_obs])
        return inv2

    def compile_mixing_net_3d(self):
            self.mix_conv1 = tf.keras.layers.Conv3D(6, [self.mix_ker,self.mix_ker,self.mix_ker],padding='same',name="mix_h1",
                                                    activation='relu',kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),data_format="channels_last")
            self.mix_conv2 = tf.keras.layers.Conv3D(1, [self.mix_ker,self.mix_ker,self.mix_ker],padding='same',name="mix_h2",
                                                    activation='relu',kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),data_format="channels_last")
            self.mix_conv3 = tf.keras.layers.Conv3D(1, [1,1,self.mix_ker], padding='same', name="mix_h3", activation=None,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                          data_format="channels_last")

    def mixing_net_3d(self,undertext,background,overtext=None,nb_obs=1):
        with tf.variable_scope("MixingNet", reuse=tf.AUTO_REUSE):
            if overtext is None:
                source_images = tf.concat([undertext, background], 3)
            else:
                source_images = tf.concat([undertext, background,overtext], 3)
            source_images = tf.reshape(source_images,shape=[-1,self.image_size,self.image_size,3,1])
            inv1 = self.mix_conv1(source_images)
            inv1 = tf.keras.layers.UpSampling3D(size = (1, 1, 2),data_format="channels_last", name="Pad1")(inv1)
            inv1 = tf.keras.layers.Dropout(0.1)(inv1)
            inv2 = self.mix_conv2(inv1)
            inv2 = tf.keras.layers.UpSampling3D(size=(1, 1, 2), data_format="channels_last",name="Pad2")(inv2)
            inv2 = tf.keras.layers.Dropout(0.1)(inv2)
            inv3 = self.mix_conv3(inv2)
            inv3 = tf.squeeze(inv3,axis=4)
            inv3 = tf.slice(inv3, [0,0,0,0],[-1,-1,-1,nb_obs])
            return inv3


    def sample_ut(self,batch_size,binary):
        if binary:
            shortcutdir = self.datadir_without_bg
        else:
            shortcutdir = self.datadir_with_bg
        org_im_list = os.listdir(shortcutdir)
        idx_list = list(range(len(org_im_list)))
        shuffle(idx_list)
        im_paths = [os.path.join(shortcutdir, org_im_list[i]) for i in idx_list[0:batch_size]]
        ims = [io.imread(fname=im_path, as_gray=True) for im_path in im_paths]
        ims = [transform.resize(im, (self.image_size, self.image_size)) for im in ims]
        ims = np.array(ims).reshape(-1, self.image_size, self.image_size, 1)
        if np.max(ims) > 1:
            print("Scale image  contrast")
            ims = ims/np.max(ims)

        if binary:
            ims = 1-ims
            ims = 2.0 * ims - 1.0
        else:
            ims = 2.0 * ims/np.max(ims) - 1.0
            for im_idx in range(len(ims)):
                contrast = np.random.uniform(0.2,1)
                ims[im_idx]= (ims[im_idx])*contrast
        return ims



    def train_disc_utb(self):
        loss_disc_utb = self.loss_dcgan_disc(self.disc_utb_real,self.disc_utb_fake)
        train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBinary/Disc")

        summ_utb = tf.summary.scalar("loss_disc_utb", loss_disc_utb)
        train_op = tf.contrib.layers.optimize_loss(loss_disc_utb, self.global_step, learning_rate=self.disc_utb_lr, optimizer='Adam',
                                                        summaries=None,
                                                        variables=train_vars)

        summary = tf.summary.merge([summ_utb] + self.summary_undertext)
        return train_op,loss_disc_utb,summary

    def train_disc_ut(self):
        loss_disc_ut = self.loss_dcgan_disc(self.disc_ut_real,self.disc_ut_fake)
        train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBack/Disc")
        train_op = tf.train.AdamOptimizer(self.disc_ut_lr).minimize(loss_disc_ut, var_list = train_vars)
        summ_ut = tf.summary.scalar("loss_disc_ut", loss_disc_ut)
        summary = tf.summary.merge([summ_ut]+self.summary_background)
        return train_op,loss_disc_ut,summary



    def train_gen_utb(self):
        loss_gen_utb = self.loss_dcgan_gen(self.disc_utb_fake)
        summary_loss = tf.summary.scalar("loss_gen_utb",loss_gen_utb)
        train_op = tf.contrib.layers.optimize_loss(loss_gen_utb, self.global_step, learning_rate=self.gen_utb_lr,
                                                   optimizer='Adam',
                                                   summaries=None,
                                                   variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TextBinary/Gen"))#["gradients"]
        summary = tf.summary.merge(self.summary_undertext+[summary_loss])
        return train_op, loss_gen_utb,summary

    def train_gen_ut(self):
        loss_gen_ut = self.loss_dcgan_gen(self.disc_ut_fake)
        antitext_loss = self.loss_anti_shortcut(self.disc_antitext_fake)
        loss = loss_gen_ut + 100.0*antitext_loss
        train_op = tf.train.AdamOptimizer(self.gen_ut_lr).minimize(loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="TextBack/Gen")+
                                                                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="MixingNet"))
        summ_loss_ut = tf.summary.scalar("loss_gen_ut",loss_gen_ut)
        summ_loss_antitext = tf.summary.scalar("loss_antitext",antitext_loss)
        summary = tf.summary.merge([summ_loss_ut,summ_loss_antitext])
        return train_op,loss_gen_ut,summary


    def loss_dcgan_gen(self,disc_fake):
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake,
            labels=tf.ones_like(disc_fake)
        ))
        return gen_cost

    def loss_anti_shortcut(self,disc_fake):
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake,
            labels=tf.zeros_like(disc_fake)
        ))
        return gen_cost

    def loss_dcgan_disc(self,disc_real,disc_fake):
        disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake,
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real,
            labels=0.75*tf.ones_like(disc_real)
        ))
        disc_cost /= 2.
        return disc_cost

    def gen_underText(self,noise=None):
        with tf.variable_scope("TextBinary/Gen",reuse=tf.AUTO_REUSE):
            output = self.gen_dc_gan(self.batch_size,z_dim=self.dim_z_utb,dim_g=self.dim_g_utb,noise=noise)
        return output

    def gen_background(self,noise=None):
        with tf.variable_scope("TextBack/Gen",reuse=tf.AUTO_REUSE):
            output = self.gen_dc_gan(self.batch_size, z_dim=self.dim_z_b, dim_g=self.dim_g_b, noise=noise)
        return output

    def disc_utb(self,input):
        dim_d = self.dim_d_utb
        with tf.variable_scope("TextBinary/Disc",reuse=tf.AUTO_REUSE):
            output = self.disc_dc_gan_undertext(input,self.batch_size,dim_d)
        return output

    def disc_ut(self,input):
        dim_d = self.dim_d_ut
        with tf.variable_scope("TextBack/Disc",reuse=tf.AUTO_REUSE):
            output = self.disc_dc_gan(input, self.batch_size, dim_d)
        return output

    def gen_dc_gan(self,batch_size,z_dim,dim_g,noise=None):
        if noise is None:
            noise = tf.random_normal([batch_size, z_dim])
        output = tf.layers.dense(noise,8 * 4 * 4 * dim_g,name='Generator.Input')
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 8 * dim_g])
        output = tf.layers.conv2d_transpose(output,8 * dim_g, 5, (2,2), padding="SAME", name='Generator.2')
        output = tf.nn.relu(output)
        output = tf.layers.conv2d_transpose(output,4 * dim_g, 5, (2,2), padding="SAME", name='Generator.3')
        output = tf.nn.relu(output)
        output = tf.layers.conv2d_transpose(output,2 * dim_g, 5, (2,2),padding="SAME", name='Generator.4')
        output = tf.nn.relu(output)
        output = tf.layers.conv2d_transpose(output,1,5,(2,2),padding="SAME", name='Generator.5')
        output = tf.nn.tanh(output)
        return output

    def disc_dc_gan(self,inputs,batch_size,dim_d):
        output = tf.layers.conv2d(inputs,dim_d,5,(2,2),padding="SAME",name='Discriminator.1')
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output, 2*dim_d, 5, (2, 2), padding="SAME", name='Discriminator.2')
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output,4*dim_d, 5, (2,2),padding="SAME", name='Discriminator.3')
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output,8*dim_d, 5, (2,2),padding="SAME", name='Discriminator.4')
        output = tf.nn.leaky_relu(output)
        output = tf.reshape(output, [-1, 8 * 4 * 4 * dim_d])
        output = tf.layers.dense(output, 1, name='Discriminator.Output' )
        return tf.reshape(output, [-1])

    def disc_dc_gan_undertext(self,inputs,batch_size,dim_d):
        output = tf.layers.conv2d(inputs,dim_d,5,(2,2),padding="SAME",name='Discriminator.1') #64->32
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output, 2*dim_d, 5, (2, 2), padding="SAME", name='Discriminator.2') #32->16
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output,4*dim_d, 5, (2,2),padding="SAME", name='Discriminator.3') #16->8
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output,8*dim_d, 5, (2,2),padding="SAME", name='Discriminator.4') #8->4
        output = tf.nn.leaky_relu(output)
        output = tf.layers.conv2d(output, 16 * dim_d, 5, (2, 2), padding="SAME", name='Discriminator.5') #4->2
        output = tf.nn.leaky_relu(output)
        output = tf.reshape(output, [-1, 8 * 4 * 4 * dim_d])
        output = tf.layers.dense(output, 1, name='Discriminator.Output' )
        return tf.reshape(output, [-1])

    def train(self,init_epoch=0):
        """
        Train model of undertext or background
         if mode - "utb" - undertext binary, trains undertext models; "ut" - undertext with back ground,
                        trains background model
        init_epoch - [default 0] if not 0 than restore model from this epoch
        """
        if self.mode=="utb":
            self.compile_model_utb()
            disc_utb_train_op, disc_utb_loss_tf, disc_utb_summ_tf = self.train_disc_utb()
            gen_utb_train_op,gen_utb_loss,gen_utb_summ = self.train_gen_utb()
            z_test = np.random.normal(0,1,[32,self.dim_z_utb])
        elif self.mode=="ut":
            self.compile_model_ut()
            disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf = self.train_disc_ut()
            gen_ut_train_op,gen_ut_loss,gen_ut_summ = self.train_gen_ut()
            z_test = np.random.normal(0, 1, [32, self.dim_z_b])

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.5)
        log_dir = os.path.join(self.main_dir,"greek_mode_{}_background_antishort_{}lutb_{}lb_" \
                  "{}gutb_{}gut_{}dut_{}dutb".format(
            self.mode,self.dim_z_utb,self.dim_z_b, self.dim_g_utb,self.dim_g_b, self.dim_d_ut,self.dim_d_utb))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        print("Working directory changed to: ", log_dir)
        writer = tf.summary.FileWriter(log_dir+ '/logs', tf.get_default_graph())
        # Train loop
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if init_epoch > 0:
                saver.restore(sess, os.path.join(log_dir, 'model-' + str(init_epoch)))

            else:
                sess.run(tf.initialize_all_variables())
                print("Initialized values")
            for iteration in range(init_epoch+1,self.nb_epochs):
                start_time = time.time()
                if self.mode == "utb":
                    _, disc_loss_utb, disc_utb_summ = sess.run([disc_utb_train_op, disc_utb_loss_tf, disc_utb_summ_tf],
                                                               feed_dict={self.utb_real: self.sample_ut(self.batch_size,
                                                                                                        binary=True)})
                    print("Epoch:{}, disc_utb loss:{}, time:{:.3f}".format(iteration, disc_loss_utb,
                                                                           time.time() - start_time))
                    writer.add_summary(disc_utb_summ, iteration)

                elif self.mode == "ut":
                    for _ in range(5):
                        _, disc_loss_ut, disc_ut_summ = sess.run([disc_ut_train_op, disc_ut_loss_tf, disc_ut_summ_tf],
                                                                 feed_dict={self.ut_real: self.sample_ut(self.batch_size, binary=False),
                                     self.utb_real: self.sample_ut(self.batch_size, binary=True)})#bool(np.random.choice([0,1],p=[0.3,0.7]))})
                        print("Epoch:{}, disc_ut loss:{}, time:{:.3f}".format(iteration,disc_loss_ut,time.time() - start_time))
                        writer.add_summary(disc_ut_summ, iteration)

                if self.mode == "utb":
                    _,gen_loss,gen_undertext_summ = sess.run([gen_utb_train_op,gen_utb_loss,gen_utb_summ],feed_dict={ self.utb_real: self.sample_ut(self.batch_size, binary=True)})
                    writer.add_summary(gen_undertext_summ,iteration)
                    if iteration%100==0:
                        saver.save(sess, os.path.join(log_dir, 'model'), global_step=iteration)
                        self.test_undertext(z_test, log_dir, iteration, True)
                elif self.mode == "ut":
                    _,gen_loss,gen_back_summ = sess.run([gen_ut_train_op,gen_ut_loss,gen_ut_summ],feed_dict={self.utb_real: self.sample_ut(self.batch_size, binary=True)})
                    writer.add_summary(gen_back_summ, iteration)
                    if iteration%50==0:
                        saver.save(sess, os.path.join(log_dir, 'model'), global_step=iteration)
        writer.close()




if __name__=="__main__":
    main_dir = r"C:/Data/PhD/bss_gan_palimpsest/training/DCGAN/test"
    datadir_without_bg = r"c:/Data/PhD/palimpsest/Greek_960/dataset_cleaned/train"
    datadir_with_bg = r"c:/Data/PhD/palimpsest/Greek_960/dataset_cleaned_with_background/train_text_background"
    mode = "utb"
    mixing_func = "mixing_net_2d"
    model = Text_Back_Gen(batch_size=128,main_dir=main_dir,mode=mode,datadir_with_bg=datadir_with_bg,datadir_without_bg=datadir_without_bg,mix_func=mixing_func)
    model.train()



