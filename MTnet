import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

import common.net as cn

class MTGenerator(cn.DCGANGenerator):
    def __init__(self, n_hidden=128, n_category=10, bottom_width=4, ch=512, wscale=0.02,
                 z_distribution="uniform", hidden_activation=F.leaky_relu, output_activation=F.tanh, use_bn=True):
        super(cn.DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.n_category = n_category
        self.ch = ch
        self.bottom_width = bottom_width
        self.z_distribution = z_distribution
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_bn = use_bn
        
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden+self.n_category, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            if self.use_bn:
                self.bn1 = L.BatchNormalization(ch // 2)
                self.bn2 = L.BatchNormalization(ch // 4)
                self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize, t, test=False):
        xp = cuda.cupy
        if test: xp.random.seed(0)
        
        if self.z_distribution == "normal":
            x = xp.random.randn(batchsize, self.n_hidden, 1, 1).astype(xp.float32)
        elif self.z_distribution == "uniform":
            x = xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(xp.float32)
        else:
            raise Exception("unknown z distribution: %s" % self.z_distribution)
            
        c = xp.array([[1 if i==t[j] else -1 for i in range(self.n_category)] for j in range(batchsize)])
        c = c.reshape(batchsize, 10, 1, 1).astype(xp.float32)
        x = F.concat((x,c), axis=1)
        return x

    def __call__(self, z):
        if not self.use_bn:
            h = F.reshape(self.hidden_activation(self.l0(z)), 
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.dc1(h))
            h = self.hidden_activation(self.dc2(h))
            h = self.hidden_activation(self.dc3(h))
            x = self.output_activation(self.dc4(h))
        else:
            h = F.reshape(self.hidden_activation(self.l0(z)), 
                          (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(self.bn1(self.dc1(h)))
            h = self.hidden_activation(self.bn2(self.dc2(h)))
            h = self.hidden_activation(self.bn3(self.dc3(h)))
            x = self.output_activation(self.dc4(h))
        return x
