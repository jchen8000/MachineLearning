import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# %matplotlib inline

"""We randomly generate some groups of data in 2-d space"""

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=600, centers=5, cluster_std=0.8, random_state=1)
plt.scatter(X[:, 0], X[:, 1], s=20);

"""Then we run the K-means algorithm, find out the centroid of each group of data, and make predictions."""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

kmeans.cluster_centers_

"""Finally we plot the data with predicted values in differnt colors, and show centroid in red circle."""

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='winter')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5);

"""## 2. K-means for image compression

https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
"""

from sklearn.datasets import load_sample_image
image = load_sample_image("china.jpg")

data = image / 255.0 # use 0...1 scale
data = data.reshape(427 * 640, 3)
print( data.shape )
reduced_colors = 16

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(reduced_colors)
kmeans.fit(data)

centers = kmeans.cluster_centers_
reduced_color_image = centers[kmeans.predict(data)]

kmeans.cluster_centers_

def plot_image_vs_reduced(image, reduced_image):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(image)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(reduced_image)
    ax[1].set_title('Reduced-color Image', size=16);
    plt.show()

image_recolored = reduced_color_image.reshape(image.shape)
plot_image_vs_reduced( image, image_recolored )

from mpl_toolkits.mplot3d import Axes3D

def plot_colors_in_colorspace( image, reduced_colors ):

    r = []
    g = []
    b = []

    for line in image:
      for pixel in line:
        temp_r, temp_g, temp_b = pixel
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)
        
    fig, ax = plt.subplots(1, figsize=(12, 10))

    ax = Axes3D(fig)
    ax.scatter(r[::16], g[::16], b[::16], cmap='viridis', lw=0, s=20, alpha=0.3, label='origian color')
    ax.scatter(reduced_colors[:, 0], reduced_colors[:, 1], reduced_colors[:, 2], c='r', lw=10, s=300, alpha=0.9, label='reduced color')
    ax.set_xlabel('R color', fontsize=20)
    ax.set_ylabel('G color', fontsize=20)
    ax.set_zlabel('B color', fontsize=20)
    ax.set_title('Original color space: 16,777,216 colors', fontsize=24)
    ax.legend(loc='lower left', fontsize=16)

    plt.show()

plot_colors_in_colorspace( image = image, reduced_colors = centers * 255 )

"""## 3. k-means for digits clustering

Here we will use k-means to identify similar digits of MNIST without using the original label .
"""

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

def plot_random_100_images(data, label, prediction = None, label_encoded = False, prediction_provided = False):
  
    print( "The small green number at the top-left is the ground truth label of the image." )
    if prediction_provided == True:
        print( "The small pink number beside it is the predicted value of the image." )
    
    _, axarr = plt.subplots(10,10,figsize=(12,12))
    plt.subplots_adjust(wspace=0.8, hspace=0.8)

    for i in range(10):
        for j in range(10):
           index = np.random.randint(data.shape[0])
           if label_encoded == False:
              groundtruth_lable = str(label.flatten()[index])
              if prediction_provided == True:
                  predicted_lable = str(prediction.flatten()[index])
           else:
              groundtruth_lable = str(np.argmax(label[index]))
              if prediction_provided == True:
                  predicted_lable = str(np.argmax(prediction[index]))
                                        
           axarr[i,j].imshow(data[index].reshape((28,28), order = 'F'), cmap="binary", interpolation="nearest")          
           axarr[i,j].axis('off')  
           axarr[i,j].text(0.5, 0.5, groundtruth_lable,bbox=dict(facecolor='lightgreen', alpha=0.5))
          
           if prediction_provided == True:
              axarr[i,j].text(10.5, 0.5, predicted_lable,bbox=dict(facecolor='pink', alpha=0.5))

              
plot_random_100_images(X_train, y_train )

X = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
y = y_train

#X = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
#y = y_test
print( X.shape)

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()
#X = sc.fit_transform(X)

kmeans = MiniBatchKMeans(10)
kmeans.fit(X)

centers = kmeans.cluster_centers_
y_pred = kmeans.predict(X)

print( "centers:", centers.shape )
print( "y_pred: ", y_pred.shape )

"""The result of K-means is 10 clusters in 784 dimensions, which is 28x28. Let's see what these cluster centers look like:"""

fig, ax = plt.subplots(1, 10, figsize=(10, 2))
centers = kmeans.cluster_centers_.reshape(10, 28, 28)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode

labels = np.zeros_like(y_pred)
for i in range(10):
    mask = (y_pred == i)
    labels[mask] = mode(y[mask])[0]

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

acs = accuracy_score(y, labels)
mat = confusion_matrix(y, labels)

print( "Accuracy: ", acs )
print( "Confusion Matrix: \n", mat )