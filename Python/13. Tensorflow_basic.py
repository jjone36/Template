# Pytorch on Colab
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip3 install torchvision

import torch
print(torch.__version__)
#################################################
####################### Basic
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

# Placeholders whose values you will specify only later
x = tf.placeholder(tf.int64, name = 'x')         # create placeholders
sigmoid = tf.sigmoid()                           # specify the computation graph
print(sess.run(sigmoid, feed_dict = {x : 3}))    # create and run the session using feed dictionary

sess.close()

####################### Simple ANN
m, n_input = features.shape
hidden_1 = 10

X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, 1])

Z1 = tf.layers.dense(X, hidden_1, activation_fn = tf.nn.relu)
A1 = tf.layers.dense(Z1, 1)

loss = tf.losses.mean_squared_error(logits = A1, labels = y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)

# Model saver
def saver():

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()
    for step in range(100):
        sess.run([optimizer, loss], feed_dict = {X : X_train, y : y_train})

    saver.save(sess, save_path, write_meta_graph = False)

#####################################################################
############# Preprocessing
im_size = 64
n_class = len(num_per_class)

images = []
labels = []

# Loading
for i in flower_labels:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)
                 if i.endswith('.jpg')]
    for f in filenames:
        img = imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

# Label encodding
def one_hot(labels):
    labels = pd.DataFrame(labels)
    labels = pd.get_dummies(labels)
    return np.array(labels)

# Shuffle
def img_shuffle(X, y, frt = .1):
    a = im_size*im_size*3
    X_train = X.reshape((X.shape[0], a))
    X_y_train = np.hstack((X_train, y))

    np.random.shuffle(X_y_train)
    cut = int(len(X_y_train) * frt)

    X_val = X_y_train[:cut, :a]
    y_val = X_y_train[:cut, a:]
    X_train = X_y_train[cut:, :a]
    y_train = X_y_train[cut:, a:]

    X_train = X_train.reshape((X_train.shape[0], im_size, im_size, 3))
    X_val = X_val.reshape((X_val.shape[0], im_size, im_size, 3))

    return X_train, X_val, y_train, y_val

images = np.array(images)
X = images.astype('float32') / 255.
labels_oh = one_hot(labels)

X_train, X_val, y_train, y_val = img_shuffle(X, labels_oh)

print("The input shape of train set is {}".format(X_train.shape))
print("The input shape of validation set is {}".format(X_val.shape))
print("The output shape of train set is {}".format(y_train.shape))
print("The output shape of validation set is {}".format(y_val.shape))

############# Modeling
# Forward Propagation
def forward_propagation(X, n_class):
    # ConvNet_1
    Z1 = tf.layers.conv2d(X, filters = 32, kernel_size = 7, strides = [2, 2], padding = 'VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # ConvNet_2
    Z2 = tf.layers.conv2d(P1, filters = 64, kernel_size = 3, strides = [1, 1], padding = 'VALID')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # Flattening
    P2 = tf.contrib.layers.flatten(P2)

    # Fully-connected
    Z3 = tf.contrib.layers.fully_connected(P2, n_class, activation_fn = None)
    return Z3

# Cost funtion
def compute_cost(y_hat, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
    return loss

# Mini-batch
def create_batch(X_train, y_train, batch_size):
    m = X_train.shape[0]
    NUM = list(np.random.permutation(m))
    X_shuffled = X_train[NUM, :]
    y_shuffled = y_train[NUM, :]

    n_batch = int(m/batch_size)
    batches = []

    for i in range(0, n_batch):
        X_batch = X_shuffled[i*batch_size:(i+1)*batch_size, :, :, :]
        y_batch = y_shuffled[i*batch_size:(i+1)*batch_size, :]

        batch = (X_batch, y_batch)
        batches.append(batch)

    X_batch_end = X_shuffled[n_batch*batch_size+1:, :, :, :]
    y_batch_end = y_shuffled[n_batch*batch_size+1:, :]
    batch = (X_batch_end, y_batch_end)
    batches.append(batch)

    return batches

# Plot the cost
def plot_cost(costs, y_hat, y):

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    pred_op = tf.argmax(y_hat, 1)
    actual = tf.argmax(y, 1)
    correct_pred = tf.equal(pred_op, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
    return accuracy

############# Computing
learning_rate = .01
epochs = 10
batch_size = 300

def model(X_train, y_train, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size):

    (m, im_size, im_size, 3) = X_train.shape
    n_class = y_train.shape[1]
    costs = []

    # Step 1. Create placeholders
    X = tf.placeholder(tf.float32, [None, im_size, im_size, 3])
    y = tf.placeholder(tf.float32, [None, n_class])

    # Step 2. Forward propagation
    y_hat = forward_propagation(X = x, n_class = n_class)

    # Step 3. Cost function
    cost = compute_cost(y_hat, y)

    # Step 4. Backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Step 5. Initialize the variables globally
    init = tf.global_variables_initializer()

    # Step 6. Run the session and compute
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # mini-batch normalization
            batches = create_batch(X_train, y_train, batch_size)
            n_batch = int(m/batch_size)

            batch_cost = 0
            for batch in batches:
                (X_batch, y_batch) = batch
                _, temp_cost = sess.run([optimizer, loss], feed_dict = {X : X_batch, y : y_batch})
                batch_cost += temp_cost/n_batch

            # Print the cost per each epoch
            if epoch % 10 == 0:
                print("Cost after {0} epoch: {1}".format(epoch, batch_cost))
            if epoch % 1 == 0:
                costs.append(batch_cost)

    # step 7. plot the cost
    acc = plot_cost(costs, y_hat = Z3, y = y_train)
    return acc
