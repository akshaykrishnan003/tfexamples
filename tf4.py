import tensorflow as tf
w= tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x=[1,2,3,4]
y=[0,-1,-2,-3]

model=w*x+b
init=tf.global_variables_initializer()
sess=tf.Session()

sq=tf.square(model-y)
loss=tf.reduce_sum(sq)
op=tf.train.GradientDescentOptimizer(0.01)
train=op.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train)
print(sess.run([w,b]))
print (float(sess.run(w)))