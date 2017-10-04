import numpy as np

import chainer
import chainer.functions as F

class Updater():
    def __init__(self, gen, opt_gen, 
                 dis, opt_dis, 
                 cls, opt_cls,
                 n_category=10, gpu=-1):
        self.gen, self.opt_gen = gen, opt_gen
        self.dis, self.opt_dis = dis, opt_dis
        self.cls, self.opt_cls = cls, opt_cls
        self.gpu = gpu
        self.n_category = n_category
        self.xp = np if gpu < 0 else chainer.cuda.cupy
        self.lam = 10.0
        self.epsilon = 100.0
        
    def update(self, x, t):
        xp = self.xp
        batchsize = x.shape[0]
        
        x_real = chainer.Variable(xp.asarray(x))
        y_real = self.dis(x_real)
        y_real_l = self.cls(x_real)
        
        loss_cls = F.softmax_cross_entropy(y_real_l, t)
        
        self.cls.cleargrads()
        loss_cls.backward()
        self.opt_cls.update()
        
        # generator
        z = self.gen.make_hidden(batchsize, t)
        x_fake = self.gen(z)
        y_fake = self.dis(x_fake)
        y_fake_l = self.cls(x_fake)
        
        loss_gen = F.sum(-y_fake) / batchsize
        loss_gen += self.epsilon * F.softmax_cross_entropy(y_fake_l, t)
        
        self.gen.cleargrads()
        loss_gen.backward()
        self.opt_gen.update()
        
        # discriminator
        x_fake.unchain_backward()
        eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[:, None, None, None]
        x_mid = eps * x_real + (1.0 - eps) * x_fake
        
        x_mid_v = chainer.Variable(x_mid.data)
        y_mid = self.dis(x_mid_v)
        dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))
        dydx = F.sqrt(F.sum(dydx ** 2, axis=(1, 2, 3)))
        loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))
        
        loss_dis = F.sum(-y_real) / batchsize
        loss_dis += F.sum(y_fake) / batchsize
        
        self.dis.cleargrads()
        loss_dis.backward()
        loss_gp.backward()
        self.opt_dis.update()
        
        return loss_gen, loss_dis, loss_cls, loss_gp
