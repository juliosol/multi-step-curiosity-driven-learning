import tensorflow as tf
from baselines.common.distributions import make_pdtype

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim


class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std

        ''' Defining variables that'll be initialized with dynamics '''
        self.dynamics = None
        self.a_samp = None
        self.entropy = None
        self.nlp_samp = None
        self.features_alt = None

        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            self.pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=False)
            self.features = unflatten_first_dim(self.flat_features, sh)

            with tf.variable_scope(scope, reuse=False):
                x = fc(self.flat_features, units=hidsize, activation=activ)
                x = fc(x, units=hidsize, activation=activ)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
                y = fc(vpred,  units=hidsize, activation=activ)
                y = fc(y, units=hidsize, activation=activ)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]

    def set_dynamics(self, dynamics):
        self.dynamics = dynamics
        with tf.variable_scope(self.scope):
            with tf.variable_scope(self.scope, reuse=False):
                ''' Changing policy to work on feature space instead of observation'''
                shaped = tf.shape(self.ph_ob)
                flat = flatten_two_dims(self.ph_ob)
                self.features_alt = self.dynamics.auxiliary_task.get_features(flat, reuse=tf.AUTO_REUSE)

                # Adding two more FC layers to more align with original architecture
                x = fc(self.features_alt, units=self.hidsize, activation=activ, reuse=True)
                x = fc(x, units=self.hidsize, activation=activ, reuse=True)
                pdparam = fc(x, name='pd', units=self.pdparamsize, activation=None)
            pdparam = unflatten_first_dim(pdparam, shaped)
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)

    def get_features(self, x, reuse):
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]
