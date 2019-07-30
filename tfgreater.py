import tensorflow as tf
is_g=tf.greater(10,9)
s=tf.Session()
print(s.run(is_g))
x=tf.to_float(is_g)
print(s.run(x))