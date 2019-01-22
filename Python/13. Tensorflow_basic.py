#################################################
############# Tensorflow Basic
import tensorflow as tf

a = tf.constant(2)      # create tensors
b = tf.constant(10)
c = tf.multiply(a, b)   # write operations between the tenseors

sess = tf.Session()     # create a Session
print(sess.run(c))      # run the session and initialize the variables


y_hat = tf.constant(36, name = 'y_hat')
y = tf.constant(39, name = 'y')
loss = tf.Variable((y - y_hat)**2, name = 'loss')

init = tf.global_variables_initializer()

with tf.Session() as sessi:
    sess.run(init)
    print(sess.run(loss))

# placeholders whose values you will specify only later
x = tf.placeholder(tf.int64, name = 'x')         # create placeholders
sigmoid = tf.sigmoid()                           # specify the computation graph
print(sess.run(sigmoid, feed_dict = {x : 3}))    # create and run the session using feed dictionary
sess.close()
