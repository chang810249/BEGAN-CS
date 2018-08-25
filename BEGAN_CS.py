#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import scipy.misc as misc
from glob import glob
from utils import *
from ops import *

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class BEGAN_CS(object):
    model_name = "BEGAN_CS"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, WGAN_para, g_lr, d_lr, result_dir, log_dir):
        self.epoch = epoch
        self.batch_size = batch_size

        self.dataset_name = dataset_name

        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        
        self.sess = sess
        self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.d_lr = tf.Variable(d_lr, name='d_lr')
        
        if dataset_name == 'celebA':
            # parameters
            self.output_height = 64
            self.output_width = 64
            self.c_dim = 3
            self.lambd = 0.25
            self.z_dim = z_dim  # dimension of noise-vector
            self.hidden_num = 128
            self.conv_block_repeat = 4

            # BEGAN-CS Parameter
            self.alpha = 0.1

            # BEGAN Parameter
            self.gamma = 0.5
            self.lamda = 0.001

            # test
            self.sample_num = 64  # number of generated images to be saved

            self.data_X = glob(os.path.join("data/CelebA/splits/train/*.jpg"))

            h, w, _ = misc.imread(self.data_X[0]).shape
            self.input_height = h
            self.input_width = w

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size     
        else:
            raise NotImplementedError

    def d_enc_block(self, x, channel_num, idx, reuse):
        with tf.variable_scope("D_enc_" + str(idx), reuse=reuse):
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < self.conv_block_repeat-1:
                channel_num2 = channel_num+self.hidden_num
            else:
                channel_num2 = channel_num
            x = slim.conv2d(x, channel_num2, 3, 1, activation_fn=tf.nn.elu)
            if idx < self.conv_block_repeat - 1:
                x = slim.conv2d(x, channel_num2, 3, 2, activation_fn=tf.nn.elu)
        return x

    def d_dec_block(self, x, idx, reuse):
        with tf.variable_scope("D_dec_" + str(idx), reuse=reuse):
            x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < self.conv_block_repeat - 1:
                x = slim.conv2d(x, self.hidden_num*2, 1, 1, activation_fn=tf.nn.elu)
                x = upscale(x, 2)
        return x

    def discriminator(self, x_, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as vs:

            # Encoder
            with tf.variable_scope("d_encoder", reuse=reuse):
                x = slim.conv2d(x_, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
                prev_channel_num = self.hidden_num
                for idx in range(self.conv_block_repeat):
                    channel_num = self.hidden_num * (idx + 1)
                    x = self.d_enc_block(x, channel_num, idx, reuse)
                x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])

            x = slim.fully_connected(x, self.z_dim, activation_fn=None)
            latent_code = x

            # Decoder
            with tf.variable_scope("d_decoder", reuse=reuse):
                num_output = int(np.prod([8, 8, self.hidden_num]))
                x = slim.fully_connected(x, num_output, activation_fn=None)
                x = reshape(x, 8, 8, self.hidden_num)
                for idx in range(self.conv_block_repeat):
                    x = self.d_dec_block(x, idx, reuse)
                    
            out = slim.conv2d(x, self.c_dim, 3, 1, activation_fn=None)

            # L1 Loss
            #recon_error = tf.reduce_mean(tf.abs(out - x_))
            
            # L2 Loss
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x_)) / self.batch_size
            d_variables = tf.contrib.framework.get_variables(vs)

            return out, recon_error, latent_code, d_variables

    def g_block(self, x, idx, reuse):
        with tf.variable_scope("g_block_" + str(idx), reuse=reuse):
            x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, self.hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < self.conv_block_repeat - 1:
                x = slim.conv2d(x, self.hidden_num*2, 1, 1, activation_fn=tf.nn.elu)
                x = upscale(x, 2)
        return x

    def generator(self, z, reuse=False):
        self.conv_block_repeat = 4

        with tf.variable_scope("generator", reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, self.hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = tf.reshape(x, [-1, 8, 8, self.hidden_num])
            for idx in range(self.conv_block_repeat):
                x = self.g_block(x, idx, reuse)
            out = slim.conv2d(x, 3, 3, 1, activation_fn=None)

        g_variables = tf.contrib.framework.get_variables(vs)
        
        return out, g_variables

    def build_model(self):
        """ BEGAN variable """
        self.k = tf.Variable(0., trainable=False)

        # images
        self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        """ Loss Function """
        # output of D for real images
        D_real_img, D_real_err, self.D_real_code, self.D_var = self.discriminator(self.inputs, reuse=False)
        
        # output of D for fake images
        G, self.G_var = self.generator(self.z, reuse=False)
        D_fake_img, D_fake_err, D_fake_code, _ = self.discriminator(G, reuse=True)

        # Latent Constraint Loss
        self.latent_constraint = self.alpha * tf.reduce_mean(tf.abs(D_fake_code - self.z))

        # get loss for discriminator
        self.d_loss = D_real_err - self.k*(D_fake_err)
        self.d_total_loss = self.d_loss + self.latent_constraint

        # get loss for generator
        self.g_loss = D_fake_err

        # convergence metric
        self.M = D_real_err + tf.abs(self.gamma*D_real_err - D_fake_err)

        # operation for updating k
        self.update_k = self.k.assign(
            tf.clip_by_value((self.k + self.lamda * (self.gamma * D_real_err - D_fake_err)), 0, 1))

        """ Training """

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.d_lr) \
                      .minimize(self.d_total_loss, var_list=self.D_var)
            self.g_optim = tf.train.AdamOptimizer(self.g_lr) \
                      .minimize(self.g_loss, var_list=self.G_var)

        """" Testing """
        # for test
        self.fake_images, _ = self.generator(self.z, reuse=True)
        self.AE_fake_images, _, self.fake_code, _ = self.discriminator(self.fake_images, reuse=True)
        self.AE_img, _, self.x_code, _ = self.discriminator(self.inputs, reuse=True)
            
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
        d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_total_loss)
        d_loss_without_constraint_sum = tf.summary.scalar("d_loss_without_constraint", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        M_sum = tf.summary.scalar("M", self.M)
        k_sum = tf.summary.scalar("k", self.k)

        latent_constraint_sum = tf.summary.scalar("latent_constraint", self.latent_constraint)

        #G_img, AE_img summary
        G_img_sum = tf.summary.image("G_images", self.fake_images)
        AE_img_sum = tf.summary.image("Reconstruct_real_images", self.AE_img)
        AE_fake_image_sum = tf.summary.image("Reconstruct_fake_images", self.AE_fake_images)
        
        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum, d_loss_without_constraint_sum])
        self.p_sum = tf.summary.merge([M_sum, k_sum, latent_constraint_sum])
        self.img_sum = tf.summary.merge([G_img_sum, AE_img_sum, AE_fake_image_sum])

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        
        # saver to save model
        t_vars = tf.all_variables()
        restore_vars = [var for var in t_vars if 'update_test' not in var.name and 'FID_Inception_Net' not in var.name]
        self.save_vars = restore_vars
        self.saver = tf.train.Saver(sharded=True, var_list=restore_vars)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.save(self.checkpoint_dir, counter)

        # loop for epoch
        start_time = time.time()
            
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_files = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                files = [get_celeba_image(batch_file,
                                          input_height=self.input_height,
                                          input_width=self.input_width,
                                          resize_height=self.output_height,
                                          resize_width=self.output_width,
                                          crop=False,
                                          grayscale=False) for batch_file in batch_files]

                batch_images = np.array(files).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_total_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update k
                _, summary_str, M_value, k_value = self.sess.run([self.update_k, self.p_sum, self.M, self.k], feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, M: %.8f, k: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, M_value, k_value))
                
                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    #fake images
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    samples = np.clip(samples, -1, 1)
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_G_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
                    #real images 
                    x_samples = self.sess.run(self.AE_img, feed_dict={self.inputs: batch_images})
                    x_samples = np.clip(x_samples, -1, 1)
                    save_images(x_samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_AE_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
                    #AE fake images
                    AE_fake_samples = self.sess.run(self.AE_img, feed_dict={self.inputs: samples})
                    AE_fake_samples = np.clip(AE_fake_samples, -1, 1)
                    save_images(AE_fake_samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_AE_fake_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            start_batch_id = 0
            self.save(self.checkpoint_dir, counter)
            
            # show temporal results
            self.visualize_results(batch_images, epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def encode(self, inputs):
        return self.sess.run(self.x_code, {self.inputs: inputs})
    
    def decode(self, z):
        return self.sess.run(self.AE_img, {self.x_code: z})

    def test(self):
        #init
        tf.global_variables_initializer().run()

        self.test_X = glob(os.path.join("data/CelebA/splits/test/*.jpg"))
        test_files = self.test_X[:self.batch_size*100]
        files = [get_celeba_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False,
                                    grayscale=False) for batch_file in test_files]
        self.val_images = np.array(files).astype(np.float32)

        t_vars = tf.all_variables()
        restore_vars = [var for var in t_vars if 'update_test' not in var.name]
        self.saver = tf.train.Saver(var_list=restore_vars)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        decoder_folder = check_folder(self.result_dir + '/' + 'real_image_decoder')
        generator_folder = check_folder(self.result_dir + '/' + 'real_image_generator')
        BEGAN_cs_folder = check_folder(self.result_dir + '/' + 'BEGAN_cs_images')

        # interpolation
        for i in range(10):
            batch_images = self.val_images[i*self.batch_size:(i+1)*self.batch_size]

            # get real images' latent code
            real_image_code = self.encode(batch_images)

            # reconstruct real images with decoder
            reoncstruct_real_image_from_decoder = self.decode(real_image_code)
            save_images(reoncstruct_real_image_from_decoder[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        decoder_folder + '/' + self.model_name + '_iter%03d' % i + '_decoder.png')

            # reconstruct real real images with G
            reconstruct_real_image_from_g = self.sess.run(self.fake_images, feed_dict={self.z: real_image_code})
            save_images(reconstruct_real_image_from_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        generator_folder + '/' + self.model_name + '_iter%03d' % i + '_generator.png')
            
            # use Decoder to interpolate real images
            batch_size = len(batch_images)
            half_batch_size = int(batch_size/2)
            iterp_D_folder_ = check_folder(decoder_folder + '/' + 'interp_D')
            iterp_D_folder = check_folder(iterp_D_folder_ + '/' + 'interp_D_iter%03d' % i)
            self.interpolate_D(batch_images[:half_batch_size], batch_images[half_batch_size:], root_path=iterp_D_folder)
            
            # use G to interpolate fake images
            interp_G_folder_ = check_folder(generator_folder + '/' + 'interp_G')
            interp_G_folder = check_folder(interp_G_folder_ + '/' + 'interp_G_iter%03d' % i)
            self.interpolate_G(batch_images, real_image_code, root_path=interp_G_folder)

            print(i)
    
    def interpolate_D(self, real1_batch, real2_batch, root_path="./results/BEGAN_ori_interpolation"):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)
        
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        decodes = np.clip(decodes, -1, 1)
        #save_image(real1_batch[:12], os.path.join(root_path, 'sstest{}_interp_D_{}.png'.format(step, 234)), nrow=10 + 2)
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            
            self.save_interp_images(img, os.path.join(root_path, 'test_interp_D_{}.png'.format(idx)), 10+2)
    
    def interpolate_G(self, real_batch, code, root_path='./results/BEGAN_ori_interpolation'):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        z = code
        z1, z2 = z[:half_batch_size], z[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.sess.run(self.fake_images, feed_dict={self.z: z})
            generated.append(z_decode)

        generated = np.clip(generated, -1, 1)
        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            self.save_interp_images(img, os.path.join(root_path, 'test_interp_G_{}.png'.format(idx)), 10+2)
    
    def save_interp_images(self, images, path, nrow):
        B, H, W, C = images.shape
        all_images = np.zeros((H, nrow*W, C))
        for i in range(nrow):
            all_images[:, i*W:(i+1)*W, :] = images[i]
        misc.imsave(path, all_images)

    def visualize_results(self, batch_images, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
        samples = np.clip(samples, -1, 1)
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_G_test_all_classes.png')
        
        x_samples = self.sess.run(self.AE_img, feed_dict={self.inputs: batch_images})
        x_samples = np.clip(x_samples, -1, 1)
        save_images(x_samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_AE_test_all_classes.png')
        
        AE_fakee_samples = self.sess.run(self.AE_img, feed_dict={self.inputs: samples})
        AE_fakee_samples = np.clip(AE_fakee_samples, -1, 1)
        save_images(AE_fakee_samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_AE_fake_test_all_classes.png')
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
