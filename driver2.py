import numpy as np
import h5py
import kmeans
import random
import os
import shutil

f = h5py.File('sift_all.h5')
sift_all = f['/sift_all'][:]

# commonly used strings to find files
base = r'C:\\Users\\niles\\Desktop\\hw1\\patches\\'
dirs = 'C:\\Users\\niles\\desktop\\hw1\\'

#each patch separated into array elements
patch_text = open('patches.txt','r')
patches = np.array(patch_text.read().split('\n'))

labels, it, centers, min_dists = kmeans.kmeans(sift_all, 1024, 'fast')

rand_clusters = []
#maybe should use issubset?
# checks for the cluster to have at least have one data point
while len(rand_clusters) < 10:
	rand_index = random.randint(0,1023)
	if rand_index in labels and rand_index not in rand_clusters:
		rand_clusters.append(rand_index)
#cluster_dirs = [dirs+str(cluster) for cluster in rand_clusters]	
for cluster in rand_clusters:
	cluster_dir = dirs + str(cluster)
	#check if dir exist already
	if os.path.exists(cluster_dir):
		# if it does then del it
		shutil.rmtree(cluster_dir)
	os.makedirs(cluster_dir)
	# returns indicies of the points that are asscociated with the cluster
	point_indexes = np.argwhere(cluster==labels).flatten()
	# gets distances of the points from cluster centroid and their indicies
	point_dist = [ (index,min_dists[index]) for index in point_indexes ]
	# sort then get the first 100 elem 
	# smallest_dist has indicies of the 100 smallest distances from the center of cluster 
	smallest_dist = sorted(point_dist,key = lambda x: x[1])[:100]
	for index in smallest_dist:
		shutil.copy(base + str(patches[index[0]]),cluster_dir)