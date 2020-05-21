#import sys
#pprint (sys.version)
#sys.exit()

import numpy as np

def pprint(*arg):
	for a in arg:
		print(a)
	return
def prnt(*arg):
	for a in arg:
		print(a)
	return

# sigmoid function
# this is used to output a probability value between 0 and 1 from any value
# we use a sigmoid function as a default model
def nonlin(x,deriv=False):
	# do we want the derivative of the value on the sigmoid curve?
	# ie. the straightline tangent at the point x on the sigmoid curve
	if(deriv==True):
		derivative = x*(1-x)
		return derivative
	# this is equal to: 1/(1+(1/e^x))
	return 1/(1+np.exp(-x))

# input dataset
# 4 samples, 3 input nodes for each
# this is arranged in the x direction (4 rows, 3 columnns)
inputData = np.array([
	[0,0,1],
	[0,1,1],
	[1,0,1],
	[1,1,1]
])
inputData = np.array([
	[30,0,200],
	[14,49,73],
	[136,189,9],
	[250,28,212]
])

# Output dataset.
# This is the desired output dataset.
# This is arranged in the x direction (4 rows, 1 column).
outputData = np.array([
	[0],
	[1],
	[1],
	[0]
])
outputData = np.array([
	[36],
	[235],
	[160],
	[9]
])

value_max = 255
value_min = 0
training_steps = 1
prnt("\nvalue_max:",value_max)
prnt("\noutputData",outputData.shape,outputData,"\ninputData",inputData.shape,inputData)
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
synapse0 = 2 * np.random.random_sample((3,4)) - 1;
# This is our second synapse, it is the mapping of the first "hidden" layer to the
#  second "hidden" layer.
# It is a matrix of dimension (4,1) because we have 4 inputs from the first hidden layer
#  and 1 output.
synapse1 = 2 * np.random.random_sample((4,1)) - 1;
prnt("\nsynapse0",synapse0.shape,synapse0,"\nsynapse1",synapse1.shape,synapse1,"\n")
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
	layer1 = nonlin(np.dot((layer0/value_max), synapse0))
	pprint("\n\nlayer1\n",layer1.shape,"\n",layer1)
	layer2 = nonlin(np.dot(layer1, synapse1))
	pprint("\n\nlayer2\n",layer2.shape,"\n",layer2)
	# How much did we miss by? How large was the error?
	# We subtract the "guesses" (layer1, our hidden layer) from the "real" answers (our output layer),
	#  which results in a vector of positive and negative numbers reflecting the difference between
	#  the guess and the real answer for each node.
	layer2_error = (outputData/value_max) - layer2

	# Output the error value calculated every 10000 steps
	if (j% 10000) == 0:
		prnt("Error:"+str(np.mean(np.abs(layer2_error))))

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
	layer2_delta = layer2_error * nonlin(layer2, True)
	pprint("\n\nlayer2_error\n", layer2_error.shape,"\n",layer2_error,"\n\nlayer2_delta\n", layer2_delta.shape,"\n",layer2_delta)

	# How much did layer1 values contribute to the layer2 error (according to the weights)?
	# We use the "confidence weighted error" from layer2 to establish an error for layer1.
	# To do this, it simply sends the error across the weights from l2 to l1.
	# This gives what you could call a "contribution weighted error" because we learn
	#  how much each node value in layer1 "contributed" to the error in layer2.
	# This step is called "backpropagating" and is the namesake of the algorithm.
	layer1_error = layer2_delta.dot(synapse1.T)
	pprint("\n\nlayer1_error\n", layer1_error.shape,"\n", layer1_error)
	layer1_delta = layer1_error * nonlin(layer1, True)
	pprint("\n\nlayer1_delta\n", layer1_delta.shape,"\n", layer1_delta)

	# Here we update the weighting of our synapses using our calculated layer1_delta.
	# We update all elements in our synapse vector (matrix of size (3,1)) at the same time since
	#  we are using "full batch" training (see line 65), but below is an example of what we are
	#  doing to each weighting/element in our synapse.
	#  Example: weight_update = input_value * layer1_delta_value
	# Since we are doing the updates in a batch operation, we must transpose the input layer, which
	#  is in the x direction (4 rows, 3 columns) to the y direction to match the direction of the
	#  layer1_delta vector (3 rows, 1 column) before we calculate the dot product and add the
	#  corrections to the synapse vector (3 rows, 1 column)
	synapse1 += layer1.T.dot(layer2_delta)
	synapse0 += layer0.T.dot(layer1_delta)
	pprint("\n\nsynapse0\n",synapse0,"\n\nsynapse1\n",synapse1)

prnt("\nDesired output:",outputData)
prnt("\nOutput after training:",layer2*value_max)
