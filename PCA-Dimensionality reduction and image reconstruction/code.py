from PIL import Image
import PIL.ImageOps
from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import math
import random
import os
from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range
%matplotlib inline
import matplotlib.pyplot as plt

N_COMPONENTS = 2
N_COMPONENTS_TO_SHOW = 2
N_DRESSES_TO_SHOW = 19
N_NEW_DRESSES_TO_CREATE = 19

# Choosing standard for image sizes here:
STANDARD_SIZE = (480,260)

def img_to_array(filename):
    """takes a filename and turns it into a numpy array of RGB pixels"""
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def makeFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# writes out each eigendress and the dresses that most and least match it
# the file names here are chosen because of the order i wanna look at the results
# (when displayed alphabetically in finder)
def createEigendressPictures():
    print("creating eigendress pictures")
    directory = "results/eigenpictures/"
    makeFolder(directory)
    for i in range(N_COMPONENTS_TO_SHOW):
        component = pca.components_[i]
        img = image_from_component_values(component)
        img.save(directory + str(i) + "_eigendress___.png")
        reverse_img = PIL.ImageOps.invert(img)
        reverse_img.save(directory + str(i) + "_eigendress_inverted.png")
        ranked_dresses = sorted(enumerate(X),
               key=lambda (a,x): x[i])
        most_i = ranked_dresses[-1][0]
        least_i = ranked_dresses[0][0]

        for j in range(N_DRESSES_TO_SHOW):
            most_j = j * -1 - 1
            Image.open(raw_data[ranked_dresses[most_j][0]][1]).save(directory + str(i) + "_eigendress__most" + str(j) + ".png")
            Image.open(raw_data[ranked_dresses[j][0]][1]).save(directory + str(i) + "_eigendress_least" + str(j) + ".png")

def indexesForImageName(imageName):
    return [i for (i,(cd,_y,f)) in enumerate(raw_data) if imageName in f]

def reconstruct(dress_number, saveName = 'reconstruct'):
    eigenvalues = X[dress_number]
    construct(eigenvalues, saveName)

def construct(eigenvalues, saveName = 'reconstruct'):
    components = pca.components_
    eigenzip = zip(eigenvalues,components)
    N = len(components[0]-400)
    r = [int(sum([w * c[i] for (w,c) in eigenzip]))
                     for i in range(N)]
    img = image_from_component_values(r)
    img.save(saveName + '.png')

def image_from_component_values(component):
    """takes one of the principal components and turns it into an image"""
    hi = max(component)
    lo = min(component)
    n = len(component) / 3
    divisor = hi - lo
    if divisor == 0:
        divisor = 1
    def rescale(x):
        return int(255 * (x - lo) / divisor)
    d = [(rescale(component[3 * i]),
          rescale(component[3 * i + 1]),
          rescale(component[3 * i + 2])) for i in range(n)]
    im = Image.new('RGB',STANDARD_SIZE)
    im.putdata(d)
    return im

def reconstructKnownDresses():
    print("reconstructing dresses...")
    directory = "results/recreatedPictures/"
    makeFolder(directory)
    for i in range(N_DRESSES_TO_SHOW):
        Image.open(raw_data[i][1]).save(directory + str(i) + "_original.png")
        saveName = directory + str(i)
        reconstruct(i, saveName)

def printComponentStatistics():
    print("component statistics:\n")
    for i in range(N_COMPONENTS_TO_SHOW):
        print("component " + str(i) + ":")
        likeComp = likesByComponent[i]
        dislikeComp = dislikesByComponent[i]
        print("means:                     like = " + str(mean(likeComp)) + "     dislike = " + str(mean(dislikeComp)))
        print("medians:                   like = " + str(median(likeComp)) + "     dislike = " + str(median(dislikeComp)))
        print("stdevs:                    like = " + str(standard_deviation(likeComp)) + "     dislike = " + str(standard_deviation(dislikeComp)))
        print("interquartile range:       like = " + str(interquartile_range(likeComp)) + "     dislike = " + str(interquartile_range(dislikeComp)))
        print("\n")



# 1. processes all photos down to a size not exceeding 512 pixels in either width or height:
print('processing images...')
print('(this takes a long time if you have a lot of images)')
lba_photos = glob('images/lba_photos/*')
process_file = img_to_array
raw_data = [(process_file(filename),filename) for filename in lba_photos]
data = np.array([cd for (cd,f) in raw_data])

# 2. using principal components analysis project your images down to a 2 dimensional representation
print('finding principal components...')
pca = PCA(n_components=N_COMPONENTS)
X = pca.fit_transform(data)
first_element = [i[0] for i in X]
second_element = [i[1] for i in X]
print 'transform complete'

# 3. visually inspect the 2D locations of each photo in the new space
plt.scatter(first_element, second_element)
plt.show()

# 4. show the reconstruction from each low-dimensional representation
reconstructKnownDresses()
print 'reconstruction complete'

# 5. finally pick a point that is far away from any known location and plot its reconstruction
far_point_2d = [15000,10000]
construct(far_point_2d, 'newpoint')
print 'new point reconstruction complete'
