#import sys
#print (sys.version)
#sys.exit()

import numpy as np

# sigmoid function
# this is used to output a probability value between 0 and 1 from any value
# we use a sigmoid function as a default model
def nonlin(x,deriv=False):
	# do we want the derivative of the value on the sigmoid curve?
	# ie. the straightline tangent at the point x on the sigmoid curve
	if(deriv==True):
		return x*(1-x)
	# this is equal to: 1/(1+(1/e^x))
	return 1/(1+np.exp(-x))

# input dataset
# 4 samples, 3 input nodes for each
# this is arranged in the x direction (4 rows, 3 columnns)
inputData = np.array([
	[0,0,1],
	[1,1,1],
	[1,0,1],
	[0,1,1]
]);
inputData = np.array([
	[[0,1,1],[1,1,1],[0,1,0]],
	[[0,1,1],[0,1,0],[1,1,1]],
	[[0,1,0],[0,1,1],[1,1,1]],
	[[0,1,0],[1,1,1],[0,1,1]]
])
# Output dataset.
# This is the desired output dataset.
# This is arranged in the y direction (1 row, 4 columns).
# T is the transpose function, this transposes this row to match our inputData
#  by rotating it to the x direction (4 rows, 1 column),
#  which means we have 3 inputs (columns above) and 1 ouput (the 1 column we have after T).
outputData = np.array([
	[0,1,1,0]
]).T
outputData = np.array([
	[[0,1,1]],
	[[0,1,0]],
	[[1,1,1]],
	[[0,0,1]]
])
print("\n\noutputData\n",outputData.shape,"\n",outputData,"\n\ninputData\n",inputData.shape,"\n",inputData)
# Seed random numbers to make calculation deterministic.
# Seeding means we will get the same "random" values each time we run this script,
#  this allows us to see how any changes to our code effect the results.
np.random.seed(1)

# Initialize weights randomly with mean average of 0.
# np.random.random_sample((3,1)) returns a 3 by 1 array of samples between 0 and 1
#  entire expression returns a 3 by 1 array of samples between -1 and 1
#  (hence "mean average of 0" above).
#
# This is our only "synapse", it is the mapping of the input layer to the middle "hidden" layer.
# It is a matrix of dimension (3,1) because we have 3 inputs and 1 output; in other words
#  in order to connect each node in the input layer (size 3) with each node in the
#  output layer (size 1), we require a matrix of dimensionality (3,1)
#
# It is best practice to initialize weights randomly with a mean average of 0.
#
# This is essentially the "neural network", all the "learning" (or modification of weights)
#  happens within this matrix since the input and output layers are transitory.
synapse0i = 2 * np.random.random_sample((4,1,3)) - 1;
synapse0j = 2 * np.random.random_sample((4,1,3)) - 1;
synapse0k = 2 * np.random.random_sample((4,1,3)) - 1;
print("\n\nsynapse0i\n",synapse0i.shape,"\n",synapse0i)
print("\n\nsynapse0j\n",synapse0j.shape,"\n",synapse0j)
print("\n\nsynapse0k\n",synapse0k.shape,"\n",synapse0k)

synapse1 = np.array([
	[1,1,1],
	[1,1,1],
	[1,1,1],
	[1,1,1]
])

# Here is where we "teach" our neural network.
# We loop through the training code 10000 times
training_steps = 60000
for j in range(training_steps):
	# Forward propagation
	# Even though our inputData contains 4 training examples (rows), we treat them as a single
	#  training example, this is called "full batch" training.
	# We can use any number of training examples using this method.
	layer0 = inputData
	##print("\n\nlayer0\n",layer0.shape,"\n",layer0)
	# This is the prediction step.
	# Here we let the neural network try to predict the output given the input.
	#
	# First we multiply layer0 (our input data) by our synapse (using dot product)
	#  then pass the output through our sigmoid function.
	# The dimensionalities are (4,3) for layer0 and (3,1) for the synapse, thus our output matrix
	#  is of dimensionality (4,1) since we must have the number of rows in the first matrix (4)
	#  and the number of columns in the second matrix (1).
	#
	# Since we gave 4 training examples, we end up with 4 guesses for the correct answer;
	#  each of these guesses is the neural network's guess for a given input.
	# The number of guesses (and dimensionality of the output matrix) will always match the number
	#  of training examples we provide (any number will work).
	#
	# (The dot product is a scalar value which is the magnitude of each vector multipled together
	#  and then by the cosine of the angle between them; this is the same as
	#  [given 2 vectors A and B] Ax*Bx + Ay*By).
	layer0i = np.empty((0,3))
	for l0i_iter in range(0,4):
		layer0i = np.append(layer0i, [layer0[l0i_iter][0]], axis=0)
		#print("\n",layer0i,"\n",layer0[l0i_iter][0])
	layer0i = layer0i.reshape(4,1,3)
	print("\n\nlayer0i\n",layer0i.shape,"\n",layer0i)
	layer1i = nonlin(np.dot(layer0i, synapse0i))
	print("\n\nlayer1i\n",layer1i.shape,"\n",layer1i)

	layer0j = np.empty((0,3))
	for l0j_iter in range(0,4):
		layer0j = np.append(layer0j, [layer0[l0j_iter][1]], axis=0)
		#print("\n",layer0i,"\n",layer0[l0i_iter][0])
	layer0j = layer0j.reshape(4,1,3)
	##print("\n\nlayer0j\n",layer0j.shape,"\n",layer0j)
	layer1j = nonlin(np.dot(layer0j, synapse0j)).reshape(4,1,3)
	##print("\n\nlayer1j\n",layer1j.shape,"\n",layer1j)

	layer0k = np.empty((0,3))
	for l0k_iter in range(0,4):
		layer0k = np.append(layer0k, [layer0[l0k_iter][2]], axis=0)
		#print("\n",layer0i,"\n",layer0[l0i_iter][0])
	layer0k = layer0k.reshape(4,1,3)
	##print("\n\nlayer0k\n",layer0k.shape,"\n",layer0k)
	layer1k = nonlin(np.dot(layer0k, synapse0k)).reshape(4,1,3)
	##print("\n\nlayer1k\n",layer1k.shape,"\n",layer1k)

	layer1 = ((layer1i + layer1j + layer1k)/3)

	##print("\n\nlayer1i\n",layer1i.shape,"\n",layer1i)
	##print("\n\nlayer1j\n",layer1j.shape,"\n",layer1j)
	##print("\n\nlayer1k\n",layer1k.shape,"\n",layer1k)
	# How much did we miss by? How large was the error?
	# We subtract the "guesses" (layer1, our hidden layer) from the "real" answers (our output layer),
	#  which results in a vector of positive and negative numbers reflecting the difference between
	#  the guess and the real answer for each node.
	layer1i_error = outputData - layer1i
	print("\n\nlayer1i_error\n", layer1i_error.shape,"\n",layer1i_error)
	layer1j_error = outputData - layer1j
	##print("\n\nlayer1j_error\n", layer1j_error.shape,"\n",layer1j_error)
	layer1k_error = outputData - layer1k
	##print("\n\nlayer1k_error\n", layer1k_error.shape,"\n",layer1k_error)
	###layer1_error = (((layer1i_error+layer1min_error)/2 + (layer1j_error+layer1min_error)/2 + (layer1k_error+layer1min_error)/2)/3).reshape(4,1,3)
	##print("\n\nlayer1_error\n", layer1_error.shape,"\n",layer1_error)

	# Output the error value calculated every 10000 steps
	if (j% 10000) == 0:
		print ("Error_i:"+str(np.mean(np.abs(layer1i_error))))
		print ("Error_j:"+str(np.mean(np.abs(layer1j_error))))
		print ("Error_k:"+str(np.mean(np.abs(layer1k_error))))
		##print ("Error:"+str(np.mean(np.abs(layer1_error))))

	# This is our "Error Weighted Derivative".
	# Here we are multiplying our error vector by the derivative for each of the guesses (which lie
	#  somewhere on our sigmoid curve).
	#
	# We use "elementwise" multiplication, which means we multiply each element in the first vector
	#  by element in the same position in the second vector.
	#  Example: [1,2,3]*[6,4,2] = [(1*6),(2*4),(3*2)] = [6,8,6]
	#
	# This reduces the error of "high confidence" predictions.
	# Guesses which have a shallow slope derivative, ie. which fall near top right or bottom left
	#  on the sigmoid curve are guesses with high confidence which need little modification.
	# Guesses which have a sharp slope, ie. which fall towards the middle of the sigmoid curve,
	#  are "wishy washy" guesses which need more modification (multiplication by greater numbers).
	layer1i_delta = layer1i_error * nonlin(layer1i, True)
	print("\n\nlayer1i_delta\n", layer1i_delta.shape,"\n",layer1i_delta)
	layer1j_delta = layer1j_error * nonlin(layer1j, True)
	##print("\n\nlayer1j_delta\n", layer1i_delta.shape,"\n",layer1j_delta)
	layer1k_delta = layer1k_error * nonlin(layer1k, True)
	##print("\n\nlayer1k_delta\n", layer1k_delta.shape,"\n",layer1k_delta)
	###layer1_delta = ((layer1i_delta.dot(layer1j_delta.T).dot(layer1k_delta))/3).reshape(4,1,3)
	##print("\n\nlayer1_delta\n", layer1_delta.shape,"\n",layer1_delta)

	# Here we update the weighting of our synapses using our calculated layer1_delta.
	# We update all elements in our synapse vector (matrix of size (3,1)) at the same time since
	#  we are using "full batch" training (see line 65), but below is an example of what we are
	#  doing to each weighting/element in our synapse.
	#  Example: weight_update = input_value * layer1_delta_value
	# Since we are doing the updates in a batch operation, we must transpose the input layer, which
	#  is in the x direction (4 rows, 3 columns) to the y direction to match the direction of the
	#  layer1_delta vector (3 rows, 1 column) before we calculate the dot product and add the
	#  corrections to the synapse vector (3 rows, 1 column)
	synapse0i += layer0i.T.dot(layer1i_delta)
	print("\n\nsynapse0i\n", synapse0i.shape,"\n",synapse0i)
	synapse0j += layer0j.T.dot(layer1j_delta)
	##print("\n\nsynapse0j\n", synapse0j.shape,"\n",synapse0j)
	synapse0k += layer0k.T.dot(layer1k_delta)
	##print("\n\nsynapse0k\n", synapse0k.shape,"\n",synapse0k)
	##synapse0 += ((synapse0i.dot(synapse0j.T).dot(synapse0k))/3)
	##print("\n\nsynapse0\n", synapse0.shape,"\n",synapse0)

print ("Output after training: ")
print("\n\nlayer1i\n",layer1i.shape,"\n",layer1i)
print("\n\nlayer1j\n",layer1j.shape,"\n",layer1j)
print("\n\nlayer1k\n",layer1k.shape,"\n",layer1k)
print ("\n\nlayer1\n",layer1.shape,"\n",layer1)
