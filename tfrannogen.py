import tensorflow as tf
for i in range(10):
    w=tf.Variable(tf.random_normal([1]))
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))