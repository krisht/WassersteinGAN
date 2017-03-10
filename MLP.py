import tensorflow as tf

def def_weight(shape, name, coll_name, reuse_scope=True):
    with tf.variable_scope('weights', reuse=reuse_scope):
        var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(coll_name, var)
        return var


def def_bias(shape, name, coll_name, reuse_scope=True):
    with tf.variable_scope('biases', reuse=reuse_scope):
        var = tf.get_variable(name=name, dtype=tf.float32, shape=shape, initializer=tf.constant_initializer(0.0))
        tf.add_to_collection(coll_name, var)
        return var

def prelu(_x, name, reuse_scope=False):
    with tf.variable_scope('prelu', reuse=reuse_scope):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        tf.add_to_collection('model_vars', alphas)

        return pos + neg

def generator(z, output_dims, input_dims):
    G_W1 = def_weight([output_dims, 128], 'g_w1', 'gen_vars', reuse_scope=False)
    G_b1 = def_bias([128], 'g_b1', 'gen_vars', reuse_scope=False)

    G_l1 = prelu(tf.matmul(z, G_W1) + G_b1, 'prelu1')

    G_W2 = def_weight([128, 256], 'g_w2', 'gen_vars', reuse_scope=False)
    G_b2 = def_bias([256], 'g_b2', 'gen_vars', reuse_scope=False)

    G_l2 = prelu(tf.matmul(G_l1, G_W2) + G_b2, 'prelu2')

    G_W3 = def_weight([256, input_dims], 'g_w3', 'gen_vars', reuse_scope=False)
    G_b3 = def_bias([input_dims], 'g_b3', 'gen_vars', reuse_scope=False)

    G_log_prob = tf.matmul(G_l2, G_W3) + G_b3
    g_prob = tf.nn.sigmoid(G_log_prob)
    return g_prob

def critic(x, input_dims, reuse_scope=False):
    D_W1 = def_weight([input_dims, 128], 'd_w1', 'crit_vars', reuse_scope=reuse_scope)
    D_b1 = def_bias([128], 'd_b1', 'crit_vars', reuse_scope=reuse_scope)

    D_l1 = prelu(tf.matmul(x, D_W1) + D_b1, 'prelu3', reuse_scope=reuse_scope)

    D_W2 = def_weight([128, 256], 'd_w2', 'crit_vars', reuse_scope=reuse_scope)
    D_b2 = def_weight([256], 'd_b2', 'crit_vars', reuse_scope=reuse_scope)

    D_l2 = prelu(tf.matmul(D_l1, D_W2) + D_b2, 'prelu4', reuse_scope=reuse_scope)

    D_W3 = def_weight([256, 1], 'd_w3', 'crit_vars', reuse_scope=reuse_scope)
    D_b3 = def_weight([1], 'd_b3', 'crit_vars', reuse_scope=reuse_scope)

    out = tf.matmul(D_l2, D_W3) + D_b3
    return out
