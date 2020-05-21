import sys
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
	[[0,0,1],[1,1,1],[0,1,0]],
	[[0,1,0],[0,1,0],[1,1,1]],
	[[1,1,1],[0,1,1],[1,1,1]],
	[[0,0,0],[1,1,1],[0,1,1]]
])
inputData = np.array([
	[[166,0,97],[108,187,5],[50,77,128]],
	[[255,0,220],[18,1,200],[100,120,7]],
	[[108,47,25],[60,225,153],[171,0,248]],
	[[149,227,231],[180,56,6],[27,9,140]]
])

# Output dataset.
# This is the desired output dataset.
# This is arranged in the x direction (4 rows, 1 column).
outputData = np.array([
	[[0,1,1]],
	[[0,1,0]],
	[[1,1,1]],
	[[0,0,1]]
])
outputData = np.array([
	[[255,1,87]],
	[[103,183,212]],
	[[9,130,0]],
	[[39,76,200]]
])
value_max = 255
training_steps = 1
print("\n\noutputData\n",outputData.shape,"\n",outputData,"\n\ninputData\n",inputData.shape,"\n",inputData)
# Seed random numbers to make calculation deterministic.
# Seeding means we will get the same "random" values each time we run this script,
#  this allows us to see how any changes to our code effect the results.
np.random.seed(1);

# Initialize weights randomly with mean average of 0.
# np.random.random_sample((3,1)) returns a 3 by 1 array of samples between 0 and 1
#  entire expression returns a 3 by 1 array of samples between -1 and 1
#  (hence "mean average of 0" above).
#
# This is our first "synapse", it is the mapping of the input layer to the first "hidden" layer.
# It is a matrix of dimension (3,4) because we have 3 inputs and 4 outputs; in other words
#  in order to connect each node in the input layer (size 3) with each node in the
#  second "hidden" layer (size 4), we require a matrix of dimensionality (3,4)
#
# It is best practice to initialize weights randomly with a mean average of 0.
#
# This is essentially the "neural network", all the "learning" (or modification of weights)
#  happens within these two matricies since the input and output layers are transitory.
synapse0i = 2 * np.random.random_sample((3,4)) - 1;
synapse0j = 2 * np.random.random_sample((3,4)) - 1;
synapse0k = 2 * np.random.random_sample((3,4)) - 1;
print("\n\nsynapse0i\n",synapse0i.shape,"\n",synapse0i)
##print("\n\nsynapse0j\n",synapse0j.shape,"\n",synapse0j)
##print("\n\nsynapse0k\n",synapse0k.shape,"\n",synapse0k)
# This is our second synapse, it is the mapping of the first "hidden" layer to the
#  second "hidden" layer.
# It is a matrix of dimension (4,1) because we have 4 inputs from the first hidden layer
#  and 1 output.
synapse1i = 2 * np.random.random_sample((4,3)) - 1;
synapse1j = 2 * np.random.random_sample((4,3)) - 1;
synapse1k = 2 * np.random.random_sample((4,3)) - 1;
print("\n\nsynapse1i\n",synapse1i.shape,"\n",synapse1i)
##print("\n\nsynapse1j\n",synapse1j.shape,"\n",synapse1j)
##print("\n\nsynapse1k\n",synapse1k.shape,"\n",synapse1k)

# Here is where we "teach" our neural network.
# We loop through the training code 10000 times
for j in range(training_steps):
	# Forward propagation
	# Even though our inputData contains 4 training examples (rows), we treat them as a single
	#  training example, this is called "full batch" training.
	# We can use any number of training examples using this method.
	layer0 = inputData

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
	##print("\n\nlayer0i\n",layer0i.shape,"\n",layer0i)
	layer1i = nonlin(np.dot((layer0i/value_max), synapse0i))

	layer0j = np.empty((0,3))
	for l0j_iter in range(0,4):
		layer0j = np.append(layer0j, [layer0[l0j_iter][1]], axis=0)
		#print("\n",layer0i,"\n",layer0[l0i_iter][0])
	##print("\n\nlayer0j\n",layer0j.shape,"\n",layer0j)
	layer1j = nonlin(np.dot((layer0j/value_max), synapse0j))

	layer0k = np.empty((0,3))
	for l0k_iter in range(0,4):
		layer0k = np.append(layer0k, [layer0[l0k_iter][2]], axis=0)
		#print("\n",layer0i,"\n",layer0[l0i_iter][0])
	##print("\n\nlayer0k\n",layer0k.shape,"\n",layer0k)
	layer1k = nonlin(np.dot((layer0k/value_max), synapse0k))

	print("\n\nlayer1i\n",layer1i.shape,"\n",layer1i)
	##print("\n\nlayer1j\n",layer1j.shape,"\n",layer1j)
	##print("\n\nlayer1k\n",layer1k.shape,"\n",layer1k)


	layer1 = np.array([layer1i,layer1j,layer1k])
	#print("\n\nlayer1\n",layer1.shape,"\n",layer1)
	#print("\n\nlayer1.T\n",layer1.T.shape,"\n",layer1.T)

	# all sublayers are (4,3)
	layer2i = nonlin(np.dot(layer1i,synapse1i)).reshape(4,1,3)
	##print("\n\nlayer2i\n",layer2i.shape,"\n",layer2i)
	layer2j = nonlin(np.dot(layer1j,synapse1j)).reshape(4,1,3)
	##print("\n\nlayer2j\n",layer2j.shape,"\n",layer2j)
	layer2k = nonlin(np.dot(layer1k,synapse1k)).reshape(4,1,3)
	##print("\n\nlayer2k\n",layer2k.shape,"\n",layer2k)
	layer2 = ((layer2i + layer2j + layer2k)/3)
	##print("\n\nlayer2\n",layer2.shape,"\n",layer2)
	# How much did we miss by? How large was the error?
	# We subtract the "guesses" (layer1, our hidden layer) from the "real" answers (our output layer),
	#  which results in a vector of positive and negative numbers reflecting the difference between
	#  the guess and the real answer for each node.
	layer2i_error = (outputData/value_max) - layer2i
	###layer2imin_error = np.sum(layer2i_error)/12
	##print("\n\nlayer2i_error\n", layer2i_error.shape,"\n",layer2i_error)
	layer2j_error = (outputData/value_max) - layer2j
	##print("\n\nlayer2j_error\n", layer2j_error.shape,"\n",layer2j_error)
	layer2k_error = (outputData/value_max) - layer2k
	##print("\n\nlayer2k_error\n", layer2k_error.shape,"\n",layer2k_error)
	###layer2_error = (((layer2i_error+layer2min_error)/2 + (layer2j_error+layer2min_error)/2 + (layer2k_error+layer2min_error)/2)/3).reshape(4,1,3)
	##print("\n\nlayer2_error\n", layer2_error.shape,"\n",layer2_error)

	# Output the error value calculated every 10000 steps
	if (j% 10000) == 0:
		print ("Error_i:"+str(np.mean(np.abs(layer2i_error))))
		print ("Error_j:"+str(np.mean(np.abs(layer2j_error))))
		print ("Error_k:"+str(np.mean(np.abs(layer2k_error))))
		##print ("Error:"+str(np.mean(np.abs(layer2_error))))

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
	layer2i_delta = layer2i_error * nonlin(layer2i, True)
	##print("\n\nlayer2i_delta\n", layer2i_delta.shape,"\n",layer2i_delta)
	layer2j_delta = layer2j_error * nonlin(layer2j, True)
	##print("\n\nlayer2j_delta\n", layer2i_delta.shape,"\n",layer2j_delta)
	layer2k_delta = layer2k_error * nonlin(layer2k, True)
	##print("\n\nlayer2k_delta\n", layer2k_delta.shape,"\n",layer2k_delta)
	###layer2_delta = ((layer2i_delta.dot(layer2j_delta.T).dot(layer2k_delta))/3).reshape(4,1,3)
	##print("\n\nlayer2_delta\n", layer2_delta.shape,"\n",layer2_delta)

	# How much did layer1 values contribute to the layer2 error (according to the weights)?
	# We use the "confidence weighted error" from layer2 to establish an error for layer1.
	# To do this, it simply sends the error across the weights from l2 to l1.
	# This gives what you could call a "contribution weighted error" because we learn
	#  how much each node value in layer1 "contributed" to the error in layer2.
	# This step is called "backpropagating" and is the namesake of the algorithm.
	layer1i_error = layer2i_delta.dot(synapse1i.T).reshape(4,4)
	##print("\n\nlayer1i_error\n", layer1i_error.shape,"\n",layer1i_error)
	layer1j_error = layer2j_delta.dot(synapse1j.T).reshape(4,4)
	##print("\n\nlayer1j_error\n", layer1j_error.shape,"\n",layer1j_error)
	layer1k_error = layer2k_delta.dot(synapse1k.T).reshape(4,4)
	##print("\n\nlayer1k_error\n", layer1k_error.shape,"\n",layer1k_error)
	###layer1_error = ((layer1i_error.dot(layer1j_error.T).dot(layer1k_error))/3)
	##print("\n\nlayer1_error\n", layer1_error.shape,"\n",layer1_error)

	layer1i_delta = (layer1i_error * nonlin(layer1i, True)).reshape(4,1,4)
	##print("\n\nlayer1i_delta\n", layer1i_delta.shape,"\n",layer1i_delta)
	layer1j_delta = (layer1j_error * nonlin(layer1j, True)).reshape(4,1,4)
	##print("\n\nlayer1j_delta\n", layer1j_delta.shape,"\n",layer1j_delta)
	layer1k_delta = (layer1k_error * nonlin(layer1k, True)).reshape(4,1,4)
	##print("\n\nlayer1k_delta\n", layer1k_delta.shape,"\n",layer1k_delta)
	###layer1_delta = ((layer1i_delta.dot(layer1j_delta.T).dot(layer1k_delta))/3)
	##print("\n\nlayer1_delta\n", layer1_delta.shape,"\n",layer1_delta)
	##print("\n\nlayer1.T\n", layer1.T.shape,"\n",layer1.T)

	# Here we update the weighting of our synapses using our calculated layer1_delta.
	# We update all elements in our synapse vector (matrix of size (3,1)) at the same time since
	#  we are using "full batch" training (see line 65), but below is an example of what we are
	#  doing to each weighting/element in our synapse.
	#  Example: weight_update = input_value * layer1_delta_value
	# Since we are doing the updates in a batch operation, we must transpose the input layer, which
	#  is in the x direction (4 rows, 3 columns) to the y direction to match the direction of the
	#  layer1_delta vector (3 rows, 1 column) before we calculate the dot product and add the
	#  corrections to the synapse vector (3 rows, 1 column)
	synapse1i += layer1i.T.dot(layer2i_delta.reshape(4,3))
	##print("\n\nsynapse1i\n", synapse1i.shape,"\n",synapse1i)
	synapse1j += layer1j.T.dot(layer2j_delta.reshape(4,3))
	##print("\n\nsynapse1j\n", synapse1j.shape,"\n",synapse1j)
	synapse1k += layer1k.T.dot(layer2k_delta.reshape(4,3))
	##print("\n\nsynapse1k\n", synapse1k.shape,"\n",synapse1k)
	##synapse1 += ((synapse1i.dot(synapse1j.T).dot(synapse1k))/3)
	##print("\n\nsynapse1\n", synapse1.shape,"\n",synapse1)


	synapse0i += layer0i.T.dot(layer1i_delta.reshape(4,4))
	##print("\n\nsynapse0i\n", synapse0i.shape,"\n",synapse0i)
	synapse0j += layer0j.T.dot(layer1j_delta.reshape(4,4))
	##print("\n\nsynapse0j\n", synapse0j.shape,"\n",synapse0j)
	synapse0k += layer0k.T.dot(layer1k_delta.reshape(4,4))
	##print("\n\nsynapse0k\n", synapse0k.shape,"\n",synapse0k)
	##synapse0 += ((synapse0i.dot(synapse0j.T).dot(synapse0k))/3)
	##print("\n\nsynapse0\n", synapse0.shape,"\n",synapse0)
print("\n\nDeired Output:\n",outputData)
print("\n\nOutput after training:\n", layer2*value_max)
