import os
import tensorflow as tf
import load_data
import model

# lambda constant for L2norm regularization (prevents overfitting)
lambda_constant = 0.001

# directory for saved model
LOGDIR = './trainedmodel'

# start tensorflow session
sess = tf.InteractiveSession()

# variables to train (weights and biases)
to_train = tf.trainable_variables()

# error function
mse = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
# include L2norm regularization in error function
loss = mse + tf.add_n([tf.nn.l2_loss(x) for x in to_train]) * lambda_constant

# use Adam algorithm to optimize weights (could be replaced by gradient descent)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

# initialize variables
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

saver = tf.train.Saver()
epochs = 30
batch_size = 100

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(load_data.data_size/batch_size)):
    images, angles = load_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: images, model.y_: angles, model.keep_prob: 0.8})
    if i % 10 == 0:
      images, angles = load_data.LoadTestBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:images, model.y_: angles, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:images, model.y_: angles, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

# to visualize the trained model in tensorboard run: "tensorboard --logdir=./logs"
# open http://0.0.0.0:6006/ in browser
