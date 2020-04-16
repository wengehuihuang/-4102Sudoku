import numpy as np
import cv2

samples = np.load('samples.npy')
labels = np.load('label.npy')

k = 80

train_input = samples[:k]
train_label = labels[:k]
test_input = samples[k:]
test_label = labels[k:]

model = cv2.ml.KNearest_create()
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)

retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
string = results.ravel()

print(test_label.reshape(1,len(test_label))[0])
print ""
print(string)