EPOCH = 10
BATHC_SIZE = 3

features, labels = (np.random.sample((100, 2)),
                   np.random.sample((100, 1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()

# a simple model
net = tf.layers.dense(x, 8, activation=tf.nn.tanh)
net = tf.layers.dense(net, 8, activation=tf.nn.tanh)
pred = tf.layers.dense(net, 1, activation=tf.nn.tanh)

loss = tf.losses.mean_squared_error(pred, y)
train_op = tf.train.AdamOptimizer().minimize(loss)

##

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCH):
        _, loss_val = sess.run([train_op, loss])
        print("Step: {}, Loss: {:.4f}".format(i, loss_val))
