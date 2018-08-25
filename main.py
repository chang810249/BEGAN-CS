import os

## GAN Variants
from BEGAN_CS import BEGAN_CS

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf
import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ('true', '1')

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='BEGAN_CS',
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='celebA',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=64, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='The learning rate of generator')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='The learning rate of discriminator')
    parser.add_argument('--train', type=str2bool, default=True)
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    models = [BEGAN_CS]

    seed = 124
    tf.set_random_seed(seed)
    np.random.seed(seed)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # declare instance for GAN
        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir,
                            g_lr=args.g_lr,
                            d_lr=args.d_lr)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        if args.train:
            gan.train()
            print(" [*] Training finished!")
        else:
            gan.test()
            print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
