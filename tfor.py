import tensorflow as tf
t,f=1.,-1.
bias=1.
train_in=[
    [t,t,bias],
    [t,f,bias],
    [f,t,bias],
    [f,f,bias]
]
train_out=[
    [t],
    [t],
    [t],
    [f]
]
w =tf.Variable(tf.zeros([3,1]))
def step(x):
    is_g=tf.greater(x,0)
    n=tf.to_float(is_g)
    d=tf.multiply(n,2)
    return tf.subtract(d,1)
output = step(tf.matmul(train_in,w))
error=tf.subtract(train_out,output)
mse=tf.reduce_mean(tf.square(error))
delta=tf.matmul(train_in,error,transpose_a=True)
train=tf.assign(w,tf.add(w,delta))
sess=tf.Session()
sess.run(tf.initialize_all_variables())
err,target=1,0
epoch,mepoch=0,10
while err>target and epoch<mepoch:
    epoch+=1
    err,_=sess.run([mse,train])
    print('epoch:',epoch,'mse:',err)