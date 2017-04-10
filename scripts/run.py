import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import numpy as np

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "trainedmodel/model.ckpt")

smoothed_angle = 0

xs = []
ys = []

#read data.txt
with open("testdata/data.txt") as f:
    for line in f:
        xs.append("testdata/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

print len(xs)
print len(ys)

squared_errors = []
for i in range(1000):
    full_image = scipy.misc.imread("testdata/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    call("clear")
    print("Image: " + str(i)+".jpg")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    print("True angle: " + str(ys[i]))
    square_error = (ys[i] - degrees)**2
    squared_errors.append(square_error)

rmse = np.sqrt(sum(squared_errors)/len(squared_errors))
print "RMSE: " + str(rmse)
