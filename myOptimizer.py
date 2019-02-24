import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


def list_devide_scalar(xs, y):
    return [x / y for x in xs]


def list_subtract(xs, ys):
    return [x - y for (x, y) in zip(xs, ys)]


def fwd_gradients(ys, xs, grad_xs=None, assert_unused=False):
    us = [array_ops.zeros_like(y) + float('nan') for y in ys]
    dydxs = tf.gradients(ys, xs, grad_ys=us)
    dydxs = [ops.convert_to_tensor(dydx) if isinstance(dydx, ops.IndexedSlices)
             else dydx for dydx in dydxs]
    if assert_unused:
        with ops.control_dependencies(dydxs):
            assert_unused = control_flow_ops.Assert(False, [1], name='fwd_gradients')
        with ops.control_dependencies([assert_unused]):
            dydxs = array_ops.identity_n(dydxs)

    dydxs = [array_ops.zeros_like(x) if dydx is None else dydx
             for x, dydx in zip(xs, dydxs)]
    for x, dydx in zip(xs, dydxs):
        dydx.set_shape(x.shape)

    dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs)

    return dysdx


def jacobian_vec(ys, xs, vs):
    # return tf.contrib.kfac.utils.fwd_gradients(ys, xs, grad_xs=vs, stop_gradients=xs)
    return fwd_gradients(ys, xs, grad_xs=vs)


def jacobian_transpose_vec(ys, xs, vs):
    dydxs = tf.gradients(ys, xs, grad_ys=vs, stop_gradients=xs)
    dydxs = [tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)]
    return dydxs


def _dot(x, y):
    dot_list = []
    for xx, yy in zip(x, y):
        dot_list.append(tf.reduce_sum(xx * yy))
    return tf.add_n(dot_list)


class SymplecticOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate, reg_params=1, use_signs=True, use_locking=True, name='symplectic_optimizer'):
        super(SymplecticOptimizer, self).__init__(use_locking=use_locking, name=name)
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self._reg_params = reg_params
        self._use_signs = use_signs

    def compute_gradients(self, loss, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        return self._gd.compute_gradients(loss, var_list,
                                          gate_gradients,
                                          aggregation_method,
                                          colocate_gradients_with_ops,
                                          grad_loss)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)
        n = len(vars_)
        h_v = jacobian_vec(grads, vars_, grads)
        ht_v = jacobian_transpose_vec(grads, vars_, grads)
        at_v = list_devide_scalar(list_subtract(ht_v, h_v), 2.)
        if self._use_signs:
            grad_dot_h = _dot(grads, ht_v)
            at_v_dot_h = _dot(at_v, ht_v)
            mult = grad_dot_h * at_v_dot_h
            lambda_ = tf.sign(mult / n + 0.1) * self._reg_params
        else:
            lambda_ = self._reg_params
        apply_vec = [(g + lambda_ * ag, x) for (g, ag, x) in zip(grads, at_v, vars_) if at_v is not None]
        return self._gd.apply_gradients(apply_vec, global_step, name)


class ConsensusOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate, beta=10, use_signs=False, use_locking=True, name='concensus_optimizer'):
        super(ConsensusOptimizer, self).__init__(use_locking=use_locking, name=name)
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self.beta = beta
        self._use_signs = use_signs

    def compute_gradients(self, loss, var_list=None, gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None):
        return self._gd.compute_gradients(loss, var_list,
                                          gate_gradients,
                                          aggregation_method,
                                          colocate_gradients_with_ops,
                                          grad_loss)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)
        n = len(vars_)

        ht_v = jacobian_transpose_vec(grads, vars_, grads)
        if self._use_signs:
            h_v = jacobian_vec(grads, vars_, grads)
            at_v = list_devide_scalar(list_subtract(ht_v, h_v), 2.)
            grad_dot_h = _dot(grads, ht_v)
            at_v_dot_h = _dot(at_v, ht_v)
            mult = grad_dot_h * at_v_dot_h
            lambda_ = tf.sign(mult / n + 0.1) * self._reg_params
        else:
            lambda_ = self.beta
        apply_vec = [(g + lambda_ * ag, x) for (g, ag, x) in zip(grads, ht_v, vars_) if ht_v is not None]
        return self._gd.apply_gradients(apply_vec, global_step, name)


class Centripetal(tf.train.Optimizer):
    def __init__(self, x, learning_rate=0.001,  beta=0.5, use_locking=False, name="Centripetal"):
        super(Centripetal, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self._og = [tf.Variable(tf.ones_like(x_)) for x_ in x]
        self._beta = beta

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)
        store_grads = [og.assign(grad,use_locking=False) for og, grad in list(zip(self._og, grads))]
        modify = [(g - og) for g, og in zip(grads, self._og)]
        apply_vec = [(g + self._beta / self._lr * mo, x) for (g, mo, x) in zip(grads, modify, vars_)]
        return control_flow_ops.group(*([self._gd.apply_gradients(apply_vec, global_step, name)]+store_grads))


class OMD(tf.train.Optimizer):
    def __init__(self, x, learning_rate=0.001,  use_locking=True, name="OMD"):
        super(OMD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self._og = [tf.Variable(tf.ones_like(x_)) for x_ in x]

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)

        modify = [(g - og) for g, og in zip(grads, self._og)]
        apply_vec = [(g + self._beta / self._lr * mo, x) for (g, mo, x) in zip(grads, modify, vars_)]
        apply_grad = [self._gd.apply_gradients(apply_vec, global_step, name)]
        with tf.control_dependencies(apply_grad):
            store_grads = [og.assign(grad, use_locking=True) for og, grad in list(zip(self._og, grads))]
        return control_flow_ops.group(*(apply_grad+store_grads))
