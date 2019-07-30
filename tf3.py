import tensorflow as tf
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
anode=a+b
s=tf.Session()
print(s.run(anode,{a:[1,3],b:[2,4]}))