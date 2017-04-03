import numpy as np

def compute_d2_slow(data, centers, D):
	# Compute the n*k matrix that contains squared Euclidean distances
	# between all data points and all centers
	# data: n*d matrix
	# centers: k*d matrix
	# D: n*k matrix (your code should rewrite this matrix)
	
	for d_row, point in enumerate(data):
		for d_col,center in enumerate(centers):
			D[d_row,d_col] = np.sum((point-center)**2)
			#works because changes data in memory location 

def compute_d2_fast(data, centers, D):
	# Compute the n*k matrix that contains squared Euclidean distances
	# using the fast algorithm based on BLAS-3 operations given in class
	# data: n*d matrix
	# centers: k*d matrix
	# D: n*k matrix (your code should rewrite this matrix)
	data_sq = data*data
	data_sums = np.sum(data_sq,axis=1).reshape(data.shape[0],1)
	centers_sq = centers*centers
	center_sums = np.sum(centers_sq,axis=1) 
	X_terms = -2*np.dot(data,centers.T)
	# creates new aray in different memory location
	# leaving D in the main program unchanged
	#D = X_terms + data_sums + center_sums # < - creates new array in diff mem location from D
	#this performs data manip at memory location of D elem so affects the D in main function
	# D[:] = X_terms + data_sums + center_sums < - works!!!
	np.copyto(D,X_terms + data_sums + center_sums) # <- this method faster than previous why? maybe using natural func than elem manip

def kmeans(data, k, alg, tol=0.00001, maxiter=100):
	# data -- n*d data matrix (each row is a data point)
	# this matrix should be in single precision in order to consume less memory

	# k -- numbers of clusters to be found

	# alg:
	# 'slow' -- using BLAS-1
	# 'fast' -- using BLAS-3

	# tol -- If the absolute difference between the cost functions of two successive iterations is less than tol, stop.
	# maxiter -- Maximum number of iterations

	# Return values:
	# labels -- a vector of cluster assignments
	# it -- number of iterations
	# centers -- cluster centroids, a k*d matrix
	# min_dists -- n-vector containing squared Euclidean distances between each data point and its closest centroid

	n, d = data.shape
	sampling = np.random.randint(0, n, k)
	centers = data[sampling, :]

	old_e = float('inf')
	old_centers = np.zeros((k, d), dtype=np.float32)
	sizes = np.zeros(k, dtype=np.uint32)

	D = np.zeros((n, k), dtype=np.float32)
	
	for it in xrange(maxiter):
		old_centers[:] = centers

		if alg == 'fast':
			compute_d2_fast(data, centers, D)
		else: # alg == 'slow'
			compute_d2_slow(data, centers, D)

		labels = np.nanargmin(D, axis=1)
		min_dists = np.nanmin(D, axis=1)
		min_dists[min_dists < 0.0] = 0.0

		centers[:, :] = 0.0
		sizes[:] = 0
		for i in xrange(n):
			assignment = labels[i]
			sizes[assignment] += 1
			centers[assignment, :] += data[i, :]

		for j in xrange(k):
			if sizes[j] > 0:
				centers[j, :] /= sizes[j]
			else:
				centers[j, :] = np.nan

		e = float(np.sqrt(np.sum(min_dists) / n))
		print 'Iteration:', it, ', Error:', e
		if it > 0 and abs(e - old_e) <= tol:
			break
		old_e = e

	return labels, it, centers, min_dists
