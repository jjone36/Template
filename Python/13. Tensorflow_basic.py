# https://github.com/jjone36/Coursera_deeplearning_ai/blob/master/course_2_Impriving%20Deep%20Neural%20Networks/5.%20Tensorflow%20Tutorial.ipynb
# https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_class2/tf_with_save.py
# https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_class/tf_example.py
# https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_class2/tensorflow2.py
#################################################
a = tf.constant(2)      # create tensors
b = tf.constant(10)
c = tf.multiply(a, b)   # write operations between the tenseors
tf.Variable()

tf.nn.conv2d()
tf.nn.batch_normalization()
tf.nn.relu()

tf.nn.max_pool()
tf.nn.avg_pool

tf.nn.contrib.layers.flatten()

tf.reduce_max()
tf.reduce_sum()
tf.exp()
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
cost = sess.run(cost, feed_dict = {z : logits, y : labels})

loss = tf.losses.mean_squared_error(logits = A1, labels = y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)
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

# Build
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


#####################################################################
############# slim
import tensorflow.contrib.slim as slim

def MyModel(images, num_classes):
    net = slim.fully_connected(input, 512, scope = 'fc1')
    logits = slim.fully_connected(net, num_classes, activation_fn = None, scope = 'fc2')
    pred = tf.nn.softmax(logits)
    return logits, pred

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(0.0001)):
    with slim.arg_scope([slim.conv2d],
                        wieghts_initializer = tf.truncated_normal_initializer(0.1),
                        activation_fn = tf.nn.relu,
                        normalizer_params = {'epsilon': .1, 'decay': .997})
        Logits = MyModel(images, num_classes, is_training = False)


#####################################################################
############# Simple ANN
m, n_input = features.shape
hidden_1 = 10

# Initialize variables
X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, 1])

# Build
Z1 = tf.layers.dense(X, hidden_1, activation_fn = tf.nn.relu)
A1 = tf.layers.dense(Z1, 1)

# Define loss, optimizer
loss = tf.losses.mean_squared_error(logits = A1, labels = y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = .01).minimize(loss)

# Start off session
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    for batch in
    saver = tf.train.Saver()
    for step in range(100):
        sess.run([optimizer, loss], feed_dict = {X : X_train, y : y_train})

    saver.save(sess, save_path, write_meta_graph = False)
