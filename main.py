import numpy as np
from myOptimizer import *
from myVisualization import kde
import os
import matplotlib.pyplot as plt
import pickle
import time

# Parameters of MLP
DEPTH = 4
WIDTH = 256
ZDIM = 32
GPU_ENABLE = True
TIME_TEST = False
OUTPUT = 'mixed_gaussian'


class MLP(tf.keras.Model):
    def __init__(self, depth, hidden_size, out_dim, name=None):
        super(MLP, self).__init__(name=name)
        self.linearLayers = []
        for i in range(depth):
            linear_layer = tf.keras.layers.Dense(hidden_size, name="hidden" + str(i))
            self.linearLayers.append(linear_layer)
        self.last_linear_layer = tf.keras.layers.Dense(out_dim, name="output")

    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.linearLayers:
            x = layer(x)
            x = tf.nn.relu(x)
        return self.last_linear_layer(x)


def x_real_builder_simple(batch_size):
    sigma = 4e-2
    skel = np.array([
        [1.5, 0.5],
        [1.5, -0.5],
        [-1.5, .5],
        [-1.5, -.5],
        [.5, 1.5],
        [.5, -1.5],
        [-.5, 1.5],
        [-.5, -1.5],
    ])
    temp = np.tile(skel, (batch_size // 4 + 1, 1))
    mus = temp[0:batch_size, :]
    return mus + sigma * tf.random_normal([batch_size, 2])


def reset_and_build_graph(learning_rate, mode, beta=None, depth=DEPTH, width=WIDTH,
                          x_real_builder=x_real_builder_simple,
                          z_dim=ZDIM, batch_size=256):
    tf.reset_default_graph()
    x_real = x_real_builder(batch_size)
    x_dim = x_real.get_shape().as_list()[1]
    generator = MLP(depth, width, x_dim, 'generator')
    discriminator = MLP(depth, width, 1, 'discriminator')
    z = tf.random_normal([batch_size, z_dim])
    x_fake = generator(z)
    disc_out_real = discriminator(x_real)
    disc_out_fake = discriminator(x_fake)

    # Loss
    disc_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_real, labels=tf.ones_like(disc_out_real)))
    disc_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_fake, labels=tf.zeros_like(disc_out_fake)))
    disc_loss = disc_loss_real + disc_loss_fake

    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_fake, labels=tf.ones_like(disc_out_fake)))
    print(tf.global_variables())
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator/')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator/')
    print(gen_vars)
    # Compute gradients
    xs = disc_vars + gen_vars
    disc_grads = tf.gradients(disc_loss, disc_vars)
    gen_grads = tf.gradients(gen_loss, gen_vars)
    Xi = disc_grads + gen_grads
    apply_vec = list(zip(Xi, xs))

    if mode == 'RMS':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif mode == 'SGA':
        optimizer = SymplecticOptimizer(learning_rate, use_signs=True)
    elif mode == 'ConOpt':
        optimizer = ConsensusOptimizer(learning_rate, use_signs=False, beta=10)
    elif mode == 'Cen':
        optimizer = Centripetal(x=xs, learning_rate=learning_rate, beta=beta)
    elif mode == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif mode == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    elif mode == 'OMD':
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    else:
        raise ValueError('reset_and_build: Mode %s not recognised' % mode)

    with tf.control_dependencies([g for (g, v) in apply_vec]):
        train_op = optimizer.apply_gradients(apply_vec)

    init = tf.global_variables_initializer()
    print("reset_and_build_graph: graph has been built.")
    return train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator


def reset_and_build_graph_alternating(learning_rate, mode, beta=None, depth=DEPTH, width=WIDTH,
                                      x_real_builder=x_real_builder_simple, z_dim=ZDIM, batch_size=256):
    tf.reset_default_graph()
    x_real = x_real_builder(batch_size)
    x_dim = x_real.get_shape().as_list()[1]
    generator = MLP(depth, width, x_dim, 'generator')
    discriminator = MLP(depth, width, 1, 'discriminator')
    z = tf.random_normal([batch_size, z_dim])
    x_fake = generator(z)
    disc_out_real = discriminator(x_real)
    disc_out_fake = discriminator(x_fake)

    # Loss
    disc_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_real, labels=tf.ones_like(disc_out_real)))
    disc_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_fake, labels=tf.zeros_like(disc_out_fake)))
    disc_loss = disc_loss_real + disc_loss_fake

    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_fake, labels=tf.ones_like(disc_out_fake)))
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator/')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator/')
    # Compute gradients
    xs = disc_vars + gen_vars
    disc_grads = tf.gradients(disc_loss, disc_vars)
    gen_grads = tf.gradients(gen_loss, gen_vars)
    Xi = disc_grads + gen_grads
    apply_vec = list(zip(Xi, xs))

    if mode == 'Cen_alt':
        print("Solver {} is called.".format(mode))
        optimizer1 = Centripetal(learning_rate=learning_rate, x=disc_vars, beta=beta)
        optimizer2 = Centripetal(learning_rate=learning_rate, x=gen_vars, beta=beta)
    elif mode == 'RMS_alt':
        print("Solver {} is called.".format(mode))
        optimizer1 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif mode == 'gd_alt':
        print("Solver {} is called.".format(mode))
        optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif mode == 'adam_alt':
        print("Solver {} is called.".format(mode))
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)

    else:
        raise ValueError('reset_and_build_graph_alternating: Mode %s not recognised' % mode)

    with tf.control_dependencies([g for (g, v) in apply_vec]):
        train_op1 = optimizer1.apply_gradients(list(zip(disc_grads, disc_vars)))
        with tf.control_dependencies([train_op1]):
            grad_y = tf.gradients(gen_loss, gen_vars)
    with tf.control_dependencies([g for (g, v) in apply_vec] + grad_y):
        train_op2 = optimizer2.apply_gradients(list(zip(gen_grads, gen_vars)))
    init = tf.global_variables_initializer()
    print("reset_and_build_graph_alternating: graph has been built.")
    return train_op2, x_fake, z, init, disc_loss, gen_loss, generator, discriminator


def train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, disciminator, z_dim=ZDIM, n_iter=20001,
          n_save=1000,
          beta=0., mode=''):
    if not os.path.isdir(OUTPUT):
        os.mkdir(OUTPUT)
    bbox = [-2, 2, -2, 2]
    batch_size = x_fake.get_shape()[0].value
    ztest = [np.random.randn(batch_size, z_dim) for i in range(10)]
    if GPU_ENABLE:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        print("train: GPU disabled")
    with tf.Session(config=config) as sess:
        sess.run(init)
        start_time = time.clock()
        time_table = []
        for i in range(n_iter):
            disc_loss_out, gen_loss_out, _ = sess.run(
                [disc_loss, gen_loss, train_op])
            if i % n_save == 0:
                print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
                      (i, disc_loss_out, gen_loss_out))
                time_table.append(time.clock() - start_time)
                if not TIME_TEST:
                    x_out = np.concatenate(
                        [sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
                    # ax = kde(x_out[:, 0], x_out[:, 1], bbox=bbox)
                    # plt.savefig('mixed_gaussian/{}_{}_{}_it_{}.png'.format(mode,DEPTH,WIDTH,i))
                if not TIME_TEST:
                    with open('{}/{}_{}_{}_it_{}.pickle'.format(OUTPUT, mode, DEPTH, WIDTH, i), 'wb') as f:
                        pickle.dump(x_out, f, pickle.HIGHEST_PROTOCOL)

        # plt.savefig('result_verify_Mom/beta{}_vertify1.png'.format(beta))
        if TIME_TEST:
            print("train: " + str(time_table))
            with open('{}/{}_{}_{}_it_time.pickle'.format(OUTPUT, mode, DEPTH, WIDTH), 'wb') as f:
                pickle.dump(time_table, f, pickle.HIGHEST_PROTOCOL)
        # generator.save_weights('test_weights/weight')
        # disciminator.save_weights('test_weights/disc_weight')


def test_one_mode(mode='RMS_alt'):
    print("test: try to use mode {}".format(mode))
    if mode == 'RMS':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=5e-4, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'RMS_alt':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph_alternating(
            learning_rate=0.0005, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'gd':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-3, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'gd_alt':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph_alternating(
            learning_rate=1e-3, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'adam':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-2, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'adam_alt':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph_alternating(
            learning_rate=1e-2, mode=mode, beta=1.)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, mode=mode)
    elif mode == 'Cen':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-3, mode=mode, beta=0.001)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=0., mode=mode)
    elif mode == 'Cen_alt':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph_alternating(
            learning_rate=5e-4, mode=mode, beta=0.5)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=0.5, mode=mode)
    elif mode == 'SGA':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-4, mode=mode, beta=1)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=1., mode=mode)
    elif mode == 'ConOpt':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-4, mode=mode, beta=1)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=1., mode=mode)
    elif mode == 'OMD':
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=1e-4, mode=mode)
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=1., mode=mode)
    else:
        raise Exception("test: The mode is not been implemented.")
    return generator, init


def test_omd():
    from shutil import copyfile
    # test_one_mode('OMD')
    learning_rate_table = [2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1., 2,
                           5., 10, 20, 50.]
    for lr in learning_rate_table:
        print("test_omd: learning_rate = {}".format(lr))
        if not os.path.exists('omd_test/lr_{}'.format(lr)):
            os.mkdir('omd_test/lr_{}'.format(lr))
        train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator = reset_and_build_graph(
            learning_rate=lr, mode='OMD')
        train(train_op, x_fake, z, init, disc_loss, gen_loss, generator, discriminator, beta=None, mode='OMD')
        for i in range(0, 20001, 1000):
            copyfile('omd_test/OMD_4_256_it_{}.pickle'.format(i), 'omd_test/lr_{}/OMD_4_256_it_{}.pickle'.format(lr, i))
    fig, axs = plt.subplots(4, 5, figsize=[3 * 5, 3 * 4])

    learning_rate_table_tile = [[2e-5, 5e-5, 1e-4, 2e-4], [5e-4, 1e-3, 2e-3, 5e-3], [1e-2, 2e-2, 5e-2, 1e-1],
                                [2e-1, 5e-1, 1., 2], [5., 10, 20., 50.]]

    for irow in range(5):
        for icol in range(4):
            lr, ax = learning_rate_table_tile[irow][icol], axs[irow][icol]
            print("plot_result: plotting data {}-{}".format(lr, 20000))
            with open('{}/lr_{}/{}_{}_{}_it_{}.pickle'.format(OUTPUT, lr, 'omd', DEPTH, WIDTH, 20000), 'rb') as f:
                x_out = pickle.load(f)
                bbox = [-2, 2, -2, 2]
                kde(x_out[:, 0], x_out[:, 1], bbox=bbox, ax=ax)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    plt.savefig('omd_table_5x4.png')


def test_all_modes():
    test_modes = ['RMS', 'RMS_alt', 'ConOpt', 'SGA', 'Cen_alt']
    for mode in test_modes:
        test_one_mode(mode)
    plot_result(['RMS', 'RMS_alt', 'ConOpt', 'SGA', 'Cen_alt'], [1000, 2000, 4000, 8000])


def test_all_time():
    global TIME_TEST
    TIME_TEST = True
    if not TIME_TEST:
        raise Exception("set TIME_TEST = True")
    test_modes = ['RMS', 'RMS_alt', 'ConOpt', 'SGA', 'Cen_alt']

    for mode in test_modes:
        test_one_mode(mode)
    plot_time_test(test_modes)


def plot_time_test(test_modes):
    time_records = dict()
    titles = ['RMSProp', 'RMSProp-alt', 'ConOpt', 'RMSProp-SGA', 'RMSProp-ACA']
    for mode in test_modes:
        with open('{}/{}_{}_{}_it_time.pickle'.format(OUTPUT, mode, DEPTH, WIDTH), 'rb') as f:
            time_records[mode] = pickle.load(f)
    print(time_records)
    barlist = []
    for key in test_modes:
        print(key)
        barlist.append(np.diff(np.array(time_records[key])).mean())
    plt.bar(x=[0,1.5,3,4.5], height=barlist[:4])
    plt.bar(x=[3, 4.5, 6], height=barlist[4])
    plt.xticks([0,1.5,3,4.5,6], titles,rotation=-10)
    plt.ylabel('sec/1000 iter')
    # plt.show()
    plt.savefig('result_time_test.png')


def plot_result(modes, iters):
    titles=['RMSProp','RMSProp-alt','ConOpt','RMSProp-SGA','RMSProp-ACA']
    fig, axs = plt.subplots(len(modes), len(iters), figsize=[3 * len(iters), 3 * len(modes)])
    for imode in range(len(modes)):
        for iiter in range(len(iters)):
            mode, iter, ax = modes[imode], iters[iiter], axs[imode][iiter]
            print("plot_result: plotting data {}-{}".format(mode, iter))
            with open('{}/{}_{}_{}_it_{}.pickle'.format(OUTPUT, mode, DEPTH, WIDTH, iter), 'rb') as f:
                x_out = pickle.load(f)
                bbox = [-2, 2, -2, 2]
                kde(x_out[:, 0], x_out[:, 1], bbox=bbox, ax=ax)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            if iiter == 0:
                print("plot_result: ylabel")
                ax.yaxis.set_label_text(titles[imode])
            if imode == len(modes) - 1:
                print("plot_result: xlabel")
                ax.set_xlabel(str(iter))
    plt.savefig('result_mode_test.png')


def plot_one_result(modes, iters):
    fig, axs = plt.subplots(len(modes), len(iters), figsize=[3 * len(iters), 3 * len(modes)])
    for imode in range(len(modes)):
        for iiter in range(len(iters)):
            mode, iter, ax = modes[imode], iters[iiter], axs[imode][iiter]
            print("plot_result: plotting data {}-{}".format(mode, iter))
            with open('{}/{}_{}_{}_it_{}.pickle'.format(OUTPUT, mode, DEPTH, WIDTH, iter), 'rb') as f:
                x_out = pickle.load(f)
                bbox = [-2, 2, -2, 2]
                kde(x_out[:, 0], x_out[:, 1], bbox=bbox, ax=ax)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            if iiter == 0:
                print("plot_result: ylabel")
                ax.yaxis.set_label_text(mode)
            if imode == len(modes) - 1:
                print("plot_result: xlabel")
                ax.set_xlabel(str(iter))
    plt.savefig('result_table.png')


if __name__ == "__main__":
    test_all_time()
    # test_all_modes()
    # plot_result(['RMS','RMS_alt','ConOpt','SGA','Cen_alt'], [1000, 2000, 4000, 8000])
    # plot_time_test(['RMS','RMS_alt','ConOpt','SGA','Cen_alt'])
    # OUTPUT = 'omd_test'
    #
    # fig, axs = plt.subplots(5, 4, figsize=[3 * 4, 3 * 5])
    # learning_rate_table_tile = [[2e-5, 5e-5, 1e-4, 2e-4], [5e-4, 1e-3, 2e-3, 5e-3], [1e-2, 2e-2, 5e-2, 1e-1],
    #                             [2e-1, 5e-1, 1., 2], [5., 10, 20, 50.]]
    #
    # for irow in range(5):
    #     for icol in range(4):
    #         lr, ax = learning_rate_table_tile[irow][icol], axs[irow][icol]
    #         print("plot_result: plotting data {}-{}".format(lr, 20000))
    #         with open('{}/lr_{}/{}_{}_{}_it_{}.pickle'.format(OUTPUT, lr, 'omd', DEPTH, WIDTH, 20000), 'rb') as f:
    #             x_out = pickle.load(f)
    #             bbox = [-2, 2, -2, 2]
    #             kde(x_out[:, 0], x_out[:, 1], bbox=bbox, ax=ax)
    #         ax.xaxis.set_ticks([])
    #         ax.yaxis.set_ticks([])
    # plt.savefig('omd_table_5x4.png')
