import os
import argparse
import numpy as np
from PIL import Image

import chainer
from chainer.dataset import convert

import MTupdater
import MTnet
import common.net as cn

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', '-t', type=int, default=1,
                        help='If negative, skip training')
    parser.add_argument('--resume', '-r', type=int, default=-1,
                        help='If positive, resume the training from snapshot')
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    flag_train = False if args.train < 0 else True
    flag_resum = False if args.resume < 0 else True
    n_epoch = args.epoch if flag_train == True else 1
    
    tsm = MTModel(args.gpu, flag_train, flag_resum, n_epoch, args.batchsize)
    tsm.run()

class MTModel():
    def __init__(self, gpu, flag_train, flag_resum, n_epoch, batchsize):
        self.n_epoch = n_epoch
        self.flag_train = flag_train
        self.flag_resum = flag_resum
        self.gpu = gpu
        self.xp = np
        self.n_hidden = 128
        self.n_category = 10
        self.sample = 10
        
        self.gen = MTnet.MTGenerator(n_hidden=self.n_hidden, n_category=self.n_category, z_distribution="uniform")
        self.dis = cn.WGANDiscriminator(output_dim=1)
        self.cls = cn.WGANDiscriminator(output_dim=self.n_category)
        
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.gen.to_gpu()
            self.dis.to_gpu()
            self.cls.to_gpu()
            self.xp = chainer.cuda.cupy
        
        self.opt_gen = chainer.optimizers.Adam()
        self.opt_gen.setup(self.gen)
        self.opt_dis = chainer.optimizers.Adam()
        self.opt_dis.setup(self.dis)
        self.opt_cls = chainer.optimizers.Adam()
        self.opt_cls.setup(self.cls)
        
        self.updater = MTupdater.Updater(self.gen, self.opt_gen,
                                         self.dis, self.opt_dis,
                                         self.cls, self.opt_cls,
                                         n_category=self.n_category, gpu=self.gpu)
        
        if self.flag_resum:
            try:
                chainer.serializers.load_npz('./net/gen.model', self.gen)
                chainer.serializers.load_npz('./net/gen.state', self.opt_gen)
                chainer.serializers.load_npz('./net/dis.model', self.dis)
                chainer.serializers.load_npz('./net/dis.state', self.opt_dis)
                chainer.serializers.load_npz('./net/cls.model', self.cls)
                chainer.serializers.load_npz('./net/cls.state', self.opt_cls)
                print('successfully resume model')
            except:
                print('ERROR: cannot resume model')
        
        train, _ = chainer.datasets.get_cifar10()
        self.N_train = len(train)
        self.train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                           repeat=True, shuffle=True)
    
    def run(self):
        xp = self.xp
        sum_loss_gen = 0
        sum_loss_dis = 0
        sum_loss_cls = 0
        sum_loss_gp = 0
        count = 0
        
        while self.train_iter.epoch < self.n_epoch:
            # train phase
            batch = self.train_iter.next()
            if self.flag_train:
                # step by step update
                x, t = convert.concat_examples(batch, self.gpu)
                loss_gen, loss_dis, loss_cls, loss_gp = self.updater.update(x,t)
                
                sum_loss_gen += float(loss_gen.data) * len(t)
                sum_loss_dis += float(loss_dis.data) * len(t)
                sum_loss_cls += float(loss_cls.data) * len(t)
                sum_loss_gp += float(loss_gp.data) * len(t)
                
            count += 1
            # test phase
            #if self.train_iter.is_new_epoch:
            if count%10 == 0:
                
                epc = self.train_iter.epoch
                thre = sum_loss_gp / self.N_train
                
                print('epoch: ', self.train_iter.epoch)
                print('G loss: {:.3f}, D loss: {:.3f}, C loss: {:.3f} gp loss: {:.3f}'.format(
                        sum_loss_gen / self.N_train,
                        sum_loss_dis / self.N_train,
                        sum_loss_cls / self.N_train,
                        sum_loss_gp / self.N_train
                        ))
                
                sum_loss_gen = 0
                sum_loss_dis = 0
                sum_loss_cls = 0
                sum_loss_gp = 0
                
                batchsize = self.n_category*self.sample
                t = xp.array([i%self.n_category for i in range(batchsize)], dtype=np.int32)
                with chainer.using_config('train', False), chainer.no_backprop_mode():
                    z = self.gen.make_hidden(batchsize, t, test=True)
                    img = self.updater.gen(z).data
                
                x = chainer.cuda.to_cpu(img)
                x = np.asarray(np.clip(x*127.5+127.5, 0.0, 255.0), dtype=np.uint8)
                
                sample_image = x.reshape((self.sample, self.n_category, 3, 32, 32)).transpose((0, 3, 1, 4, 2))
                sample_image = sample_image.reshape((32*self.sample, 32*self.n_category, 3))
                Image.fromarray(sample_image).save('./img/sample.png')
                
            if self.train_iter.is_new_epoch:
                try:
                    epc = self.train_iter.epoch
                    
                    chainer.serializers.save_npz('./net/gen' + str(epc) + '.model', self.gen)
                    chainer.serializers.save_npz('./net/gen' + str(epc) + '.state', self.opt_gen)
                    chainer.serializers.save_npz('./net/dis' + str(epc) + '.model', self.dis)
                    chainer.serializers.save_npz('./net/dis' + str(epc) + '.state', self.opt_dis)
                    chainer.serializers.save_npz('./net/cls.model', self.cls)
                    chainer.serializers.save_npz('./net/cls.state', self.opt_cls)
                    
                    if epc > 1:
                        os.remove('./net/gen' + str(epc-1) + '.model')
                        os.remove('./net/gen' + str(epc-1) + '.state')
                        os.remove('./net/dis' + str(epc-1) + '.model')
                        os.remove('./net/dis' + str(epc-1) + '.state')
                    
                    print('Successfully saved model')
                except:
                    print('ERROR: saving model ignored')
        
if __name__ == '__main__':
    main()
