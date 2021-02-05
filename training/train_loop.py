import tensorflow as tf
import time
import os
import numpy as np
import cv2

import utils

from training import model
from training import loss


def get_grid_imgs(G, batch_size, z_dim, z_test=None):    
    g_imgs = []
    if 16 > batch_size:        
        assert 16 % batch_size == 0, 'batch size is not 2^n'
        for i in range(16//batch_size):
            if z_test is None:
                z = np.random.normal( size=[batch_size, z_dim] )
            else:
                z = z_test[i*batch_size:(i+1)*batch_size]
            fake_imgs = G(z)
            g_imgs.append(fake_imgs)
    else:
        if z_test is None:
            z = np.random.normal( size=[batch_size, z_dim] )
        else:
            z = z_test
        fake_imgs = G(z)[:16]
        g_imgs.append(fake_imgs)
        

    g_imgs = np.array(g_imgs)
    group, minibatch, h, w, c = g_imgs.shape
    b = group*minibatch

    g_imgs = np.reshape(g_imgs, (b, h, w, c))

    rows = 4
    cols = 4
    g_imgs = ((g_imgs+1)*127.5).astype(np.uint8)

    grid_img = np.zeros((h*rows, w*cols, c), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = g_imgs[row*cols + col]

    return grid_img


def train(dataset_dir, result_dir, batch_size, epochs, lr, load_path, save_term):
    # make results dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    files = os.listdir(result_dir)
    results_number = len(files)

    #set dir name
    desc = 'DCGAN'
    desc +='_batch-%d'%batch_size
    desc +='_epoch-%d'%epochs

    save_dir = os.path.join(result_dir, '%04d_' %(results_number) + desc)
    os.mkdir(save_dir)

    #ckpt dir
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    os.mkdir(ckpt_dir)

    # set my logger
    log = utils.my_log(os.path.join(save_dir, 'results.txt'))
    log.logging('< Info >')


    # load data
    imgs = utils.load_images_in_folder(dataset_dir)
    imgs_num = imgs.shape[0]
    log.logging('dataset path : ' + dataset_dir)
    log.logging('results path : '+ save_dir)
    log.logging('load model path : '+ str(load_path))
    log.logging('load images num : %d' %(imgs_num))
    log.logging('image shape : (%d, %d, %d)' %(imgs.shape[1], imgs.shape[2], imgs.shape[3]))

    
    # images preprocessing [-1 , 1]
    imgs = (imgs - 127.5)/127.5
    

    ### train setting
    np.random.seed(2222)
    z_dim = 100
    beta_1 = 0.5
    log.logging('z dim : %d' %z_dim)

    # input placeholder
    g_in = tf.keras.layers.Input(shape=(z_dim), name='G_input')
    d_in = tf.keras.layers.Input(shape=(imgs.shape[1],imgs.shape[2],imgs.shape[3]), name='D_input')

    y_true = tf.keras.layers.Input(shape=(1), name='y_true')

    # set model

    G = tf.keras.Model(g_in, model.Generator(g_in), name='Generator')
    D = tf.keras.Model(d_in, model.Discriminator(d_in), name='Discriminator')
    GAN = tf.keras.Model(g_in, D(G(g_in)), name='DCGAN')

    #G.summary()
    #D.summary()
    #GAN.summary()

    # set optimizer
    G_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, name='Adam_G')
    D_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, name='Adam_D')

    log.logging('G opt : Adam(lr:%f, beta_1:%f)' %(lr, beta_1))
    log.logging('D opt : Adam(lr:%f, beta_1:%f)' %(lr, beta_1))
    log.logging('total epoch : %d' %epochs)
    log.logging('batch size  : %d' %batch_size)
    
    log.logging(utils.SPLIT_LINE, log_only=True)

    # train 
    log.logging('< train >')

    train_start_time = time.time()

    # test noise z
    z_test = np.random.normal(size=(batch_size, z_dim))

    # load model
    if load_path is not None:
        GAN.load_weights(load_path)
        log.logging('['+load_path+'] model loaded !!')

    for epoch in range(1, epochs+1):
        start_epoch_time = time.time()
        epoch_g_loss = 0
        epoch_d_loss = 0
        # remaining data is not used
        for step in range(imgs_num//batch_size): 
                        
            # training D
            with tf.GradientTape() as tape_fake, tf.GradientTape() as tape_real:
                z = np.random.normal( size=[batch_size, z_dim] )
                real_imgs = imgs[step*batch_size:(step+1)*batch_size]
                fake_imgs = G(z, training=True)

                fake_logits = D(fake_imgs, training=True)
                real_logits = D(real_imgs, training=True)

                d_loss_fake = loss.loss(np.zeros((batch_size, 1)), fake_logits)
                d_loss_real = loss.loss(np.ones((batch_size, 1)), real_logits)

            d_gradient_fake = tape_fake.gradient(d_loss_fake, D.trainable_variables)
            d_gradient_real = tape_real.gradient(d_loss_real, D.trainable_variables)
            D_opt.apply_gradients(zip(d_gradient_fake, D.trainable_variables))
            D_opt.apply_gradients(zip(d_gradient_real, D.trainable_variables))
            
            # training G
            with tf.GradientTape() as tape:                
                z = np.random.normal( size=[batch_size, z_dim] )
                fake_imgs = G(z, training=True)
                g_logits = D(fake_imgs, training=True)
                g_loss = loss.loss(np.ones((batch_size, 1)), g_logits)

            g_gradient = tape.gradient(g_loss, G.trainable_variables)
            G_opt.apply_gradients(zip(g_gradient, G.trainable_variables))

            # calculate loss
            loss_g = g_loss

            loss_d = d_loss_real + d_loss_fake

            epoch_g_loss += loss_g
            epoch_d_loss += loss_d

            print('%03d / %03d loss (G : %f || D : %f ) detail:(f:%f r:%f)' %(step, imgs_num//batch_size, loss_g, loss_d, d_loss_fake, d_loss_real) )
        
        epoch_g_loss /= (imgs_num//batch_size) 
        epoch_d_loss /= (imgs_num//batch_size) 
        
        log.logging('[%d/%d] epoch << G loss: %.5f || D loss: %.5f >>  time: %.1f sec' %(epoch, epochs, epoch_g_loss, epoch_d_loss, time.time()-start_epoch_time))

        # make fake imgs
        fake_imgs = get_grid_imgs(G, batch_size, z_dim, z_test)
        cv2.imwrite(os.path.join(save_dir, 'fake %05depoc.png' %epoch), fake_imgs)

        # model save
        if epoch % save_term == 0:
            save_name = os.path.join(ckpt_dir, 'model-%d.h5' %(epoch))
            GAN.save(save_name)
    
    log.logging('\n[%d] epoch finish! << fianl G loss: %.5f || final D loss: %.5f >>  total time: %.1f sec' %(epochs, epoch_g_loss, epoch_d_loss, time.time()-train_start_time))


    save_name = os.path.join(save_dir, 'model-%d.h5' %(epoch))
    GAN.save(save_name)
    print('train finished!')


def validation(load_path, result_dir, generate_num, seed):
    # make results dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    files = os.listdir(result_dir)
    results_number = len(files)

    #set dir name
    desc = 'DCGAN'
    desc +='_generate'
    save_dir = os.path.join(result_dir, '%04d_' %(results_number) + desc)
    os.mkdir(save_dir)

    # set my logger
    log = utils.my_log(os.path.join(save_dir, 'results.txt'))
    log.logging(utils.SPLIT_LINE)
    log.logging('< Validation >')

    # set seed
    np.random.seed(seed) #default 22222

    # validation set 
    z_dim = 100

    # input placeholder
    g_in = tf.keras.layers.Input(shape=(z_dim), name='G_input')

    # set model    
    GAN = tf.keras.models.load_model(load_path)
    G = GAN.layers[1]
    log.logging('['+load_path+'] model loaded !!')


    for idx in range(generate_num):
        z = np.random.normal( size=[1, z_dim] )
        fake_img =  G.predict(z)[0]

        # pixel range : -1 ~ 1 -> 0 ~ 255
        fake_img = ((fake_img + 1) * 127.5).astype(np.uint8)
        log.logging('[%d/%d] Generate image!! [seed:%d]' %(idx, generate_num, seed))
        cv2.imwrite(os.path.join(save_dir, 'fake%04d_seed%d.png' %(idx, seed)), fake_img)

        



