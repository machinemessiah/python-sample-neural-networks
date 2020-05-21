import sys

import numpy as np

def pprint(*arg):
	return
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
		return x*(1-x)
	# this is equal to: 1/(1+(1/e^x))
	return 1/(1+np.exp(-x))

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
training_steps = 20000
num_layers = 8
output_shape = outputData.shape
input_shape = inputData.shape
structure_width = output_shape[len(output_shape)-1]
structure_height = output_shape[0]
structure_samples = input_shape[len(input_shape)-2]
input_sample_width = input_shape[len(input_shape)-1]
pprint("\nstructure_width",structure_width,"\nstructure_height",structure_height,"\nstructure_samples",structure_samples)

np.random.seed(1);

# Kick out if the input data is not suitable for the output
if (len(output_shape) != len(input_shape)) :
	sys.exit("inputData's shape must be of the same length as the outputData's shape")

# Create synapses based on the number of layers and the dimenion of the input and output
last_output = num_layers+1
synapses = {}
for l in range(0,num_layers):
	if (l == 0) :
		for s in range(0,structure_samples):
			synapses[str(l)+"_"+str(s)] = 2 * np.random.random_sample((structure_width,last_output)) - 1
	elif (l == (num_layers-1)) :
		for s in range(0,structure_samples):
			synapses[str(l)+"_"+str(s)] = 2 * np.random.random_sample((structure_height,structure_width)) - 1
	else :
		for s in range(0,structure_samples):
			synapses[str(l)+"_"+str(s)] = 2 * np.random.random_sample((last_output,last_output-1)) - 1
		last_output -= 1

for j in range(training_steps):

	layer0 = inputData

	# Create the layers
	layers = {}
	for l in range(0, num_layers):
		if (l == 0) :
			for s in range(0,structure_samples):
				layers[str(l)+"_"+str(s)] = np.empty((0,input_sample_width))
				for i in range(0,input_sample_width+1):
					layers[str(l)+"_"+str(s)] = np.append(layers[str(l)+"_"+str(s)], [layer0[i][0]], axis=0)
					pprint("\nlayer"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)
		elif (l == 1) :
			for s in range(0,structure_samples):
				layers[str(l)+"_"+str(s)] = nonlin(np.dot((layers[str(l-1)+"_"+str(s)]/value_max), synapses[str(l-1)+"_"+str(s)]))
				pprint("\nlayer"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)
		elif (l == (num_layers-1)) :
			output_layer = 0
			for s in range(0,structure_samples):
				layers[str(l)+"_"+str(s)] = nonlin(np.dot(layers[str(l-1)+"_"+str(s)],synapses[str(l-1)+"_"+str(s)])).reshape(output_shape)
				output_layer += layers[str(l)+"_"+str(s)]
				pprint("\nlayer"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)
			output_layer = output_layer/structure_samples
		else :
			for s in range(0,structure_samples):
				layers[str(l)+"_"+str(s)] = nonlin(np.dot(layers[str(l-1)+"_"+str(s)],synapses[str(l-1)+"_"+str(s)]))
				pprint("\nlayer"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)

	# Create errors and deltas
	errors = {}
	deltas = {}
	for l in range(num_layers-1, 0, -1):
		if (l == (num_layers-1)) :
			for s in range(0,structure_samples):
				errors[str(l)+"_"+str(s)] = (outputData/value_max) - layers[str(l)+"_"+str(s)]
				if (j% 10000) == 0:
					prnt ("Error-"+str(l)+"_"+str(s)+":"+str(np.mean(np.abs(errors[str(l)+"_"+str(s)]))))
				deltas[str(l)+"_"+str(s)] = errors[str(l)+"_"+str(s)] * nonlin(layers[str(l)+"_"+str(s)], True)
				pprint("\ndelta"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)
		else :
			for s in range(0,structure_samples):
				errors[str(l)+"_"+str(s)] = deltas[str(l+1)+"_"+str(s)].dot(synapses[str(l)+"_"+str(s)].T)
				reshape = (errors[str(l)+"_"+str(s)].shape[0], errors[str(l)+"_"+str(s)].shape[len(errors[str(l)+"_"+str(s)].shape)-1])
				errors[str(l)+"_"+str(s)] = errors[str(l)+"_"+str(s)].reshape(reshape)
				deltas[str(l)+"_"+str(s)] = (errors[str(l)+"_"+str(s)] * nonlin(layers[str(l)+"_"+str(s)], True))
				pprint("\ndelta"+str(l)+"_"+str(s),layers[str(l)+"_"+str(s)].shape)

	# Update synapses
	last_output = num_layers+1
	for l in range(0,num_layers-1):
		if (l == (num_layers-2)) :
			for s in range(0,structure_samples):
				synapses[str(l)+"_"+str(s)] += layers[str(l)+"_"+str(s)].T.dot(deltas[str(l+1)+"_"+str(s)].reshape(structure_height,structure_width))
		else :
			for s in range(0,structure_samples):
				synapses[str(l)+"_"+str(s)] += layers[str(l)+"_"+str(s)].T.dot(deltas[str(l+1)+"_"+str(s)])


prnt("\nDesired Output:",outputData)
prnt("\nOutput after training:", output_layer*value_max)
