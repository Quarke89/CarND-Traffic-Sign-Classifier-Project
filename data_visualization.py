# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'dataset/train.p'
validation_file= 'dataset/valid.p'
testing_file = 'dataset/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


with open("signnames.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    data = [r for r in reader]

fig, axes = plt.subplots(nrows=9, ncols=5, figsize=(10,10))
fig.tight_layout()
for n in range(43):	
	item_index = np.where(y_train==n)	
	# print(axes[n])
	axes[np.unravel_index(n, (9,5))].imshow(X_train[item_index[0][1],:,:,:])
	axes[np.unravel_index(n, (9,5))].set_title(data[n][1])
	axes[np.unravel_index(n, (9,5))].set_axis_off()

axes[np.unravel_index(43, (9,5))].set_axis_off()
axes[np.unravel_index(44, (9,5))].set_axis_off()
plt.show()


