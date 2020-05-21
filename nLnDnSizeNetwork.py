import sys

import numpy as np

import json

class swapable_iterator(object):
	def __init__(self, it):
		self.it = iter(it)
		self.object = it
		self.previous_object = self.object

	def __iter__(self):
		return self

	def next(self):
		return next(self.it)

	__next__ = next # Python 3 compatibility

	def swap(self, new_it):
		self.previous_object = self.object
		self.it = iter(new_it)
		self.object = new_it

def pprint(*arg):
	#return
	for a in arg:
		print(a)
	return

def prnt(*arg):
	for a in arg:
		print(a)
	return

def number_to_base(n, b):
	if n == 0:
		return [0]
	digits = []
	while n:
		digits.append(int(n % b))
		n //= b
	return digits[::-1]

def deep_access(x, keylist):
     val = x
     for key in keylist:
         val = val[key]
     return val


# sigmoid function
# this is used to output a probability value between 0 and 1 from any value
# we use a sigmoid function as a default model
def nonlin(x, deriv=False):
	# do we want the derivative of the value on the sigmoid curve?
	# ie. the straightline tangent at the point x on the sigmoid curve
	if(deriv==True):
		return x*(1-x)
	# this is equal to: 1/(1+(1/e^x))
	return 1/(1+np.exp(-x))

def chunk_output():
	chunked_output = {}
	for e in range(0,grouping_size):
		index = generate_name(0,e)
		chunked_output[e] = np.empty((0,input_sample_width))
		for i in range(0,structure_height):
			access_index = [int(n) for n in index.split("_")]
			access_index[0] = i
			chunked_output[e] = np.append(chunked_output[e], [deep_access(outputData,access_index)], axis=0)
	return chunked_output


def find_sample_shape():
	sample_shape = {0:0}
	located_sample = 0
	i = 1
	for dim in output_shape:
		if (located_sample == 1):
			sample_shape[i] = dim
			i += 1
		elif (dim == 1):
			located_sample = 1
	return tuple(sample_shape.values())

def remove_empty_dimensions(shape):
	new_shape = {}
	i = 0
	for dim in shape:
		if (dim != 1):
			new_shape[i] = dim
			i += 1
	return tuple(new_shape.values())

def generate_name(layer, input_layer):
	skip_size = structure_samples - grouping_size
	skip_counter = 1
	for x in range(1, input_layer+2):
		if (skip_counter != 0 and skip_counter % structure_samples == 0):
			input_layer += skip_size
			skip_counter = 1
		skip_counter += 1
	name = "".join(str(x) for x in number_to_base(input_layer,structure_samples))
	while len(name) < grouping_size:
		name = "0"+name
	name = "_".join(list(name))
	return str(layer)+"_"+name

def create_synapses():
	last_output = num_layers+1
	synapses = {}
	for l in range(0,num_layers-1):
		if (l == 0):
			for input_layer in range(0,num_subelements):
				# Generate the synapse name
				synapse_name = generate_name(l,input_layer)
				synapses[synapse_name] = 2 * np.random.random_sample((structure_width,last_output)) - 1
			pprint("synapse"+synapse_name,synapses[synapse_name].shape)
		elif (l == (num_layers-1)):
			for input_layer in range(0,num_subelements):
				# Generate the synapse name
				synapse_name = generate_name(l,input_layer)
				synapses[synapse_name] = 2 * np.random.random_sample((structure_height,structure_width)) - 1
			pprint("synapse"+synapse_name,synapses[synapse_name].shape)
		else :
			for input_layer in range(0,num_subelements):
				# Generate the synapse name
				synapse_name = generate_name(l,input_layer)
				synapses[synapse_name] = 2 * np.random.random_sample((last_output,last_output-1)) - 1
			pprint("synapse"+synapse_name,synapses[synapse_name].shape)
			last_output -= 1
	return synapses

def generate_output(output_layers):
	shape_array = [0] + list(input_sample_shape)
	rows = np.empty(shape_array)
	prnt(output_layers)
	for s in range(0,structure_height-1):
		column_number = 1
		column = np.empty(input_sample_shape)
		target = swapable_iterator(output_layers[s])
		prnt(target.object.shape,target.object)
		for sample in target:
			if sample.shape == input_sample_shape[1:]:
				# we have a sample we're looking for
				column += sample
				column_number += 1
			else:
				prnt(sample)
				target.swap(sample)

		# divide column by number of columns added together to make it up
		column = column/column_number
		# append the column to rows
		rows = np.append(rows,[column])
	prnt(rows.shape,rows)


def generate_output_layer():
	output_layer = np.empty(output_shape)
	output_chunks = swapable_iterator(output_layers)
	chunks = {}
	chunk_number = 0
	subchunk_number = 0
	current_chunk = {}
	for output_chunk in output_chunks:
		for subchunk_width in input_sample_shape[1:-1]:
			if subchunk_number <= (subchunk_width - 1):
				current_chunk[subchunk_number] = output_layers[(subchunk_width*chunk_number)+subchunk_number]
				subchunk_number += 1
			else:
				chunks[chunk_number] = current_chunk
				prnt("\nCHUNK"+str(chunk_number),current_chunk)
				current_chunk = {}
				chunk_number += 1
				subchunk_number = 0

	return chunks
	#output_layers = output_layers.reshape(output_shape)
	#return output_layers*value_max


def create_layers():
	layer0 = inputData

	# Create the layers
	layers = {}
	for l in range(0, num_layers):
		if (l == 0):
			for layer in range(0,num_subelements):
				layer_name = generate_name(l,layer)
				layers[layer_name] = np.empty((0,input_sample_width))
				for i in range(0,structure_height):
					access_index = [int(n) for n in layer_name.split("_")]
					access_index[0] = i
					layers[layer_name] = np.append(layers[layer_name], [deep_access(layer0,access_index)], axis=0)
			pprint("layer"+layer_name,layers[layer_name].shape)
		elif (l == 1):
			for layer in range(0,num_subelements):
				layer_name =  generate_name(l,layer)
				previous_layer_name =  generate_name(l-1,layer)
				layers[layer_name] = nonlin(np.dot((layers[previous_layer_name]/value_max), synapses[previous_layer_name]))
			pprint("layer"+layer_name,layers[layer_name].shape)
		elif (l == (num_layers-1)):
			for layer in range(0,num_subelements):
				layer_name =  generate_name(l,layer)
				previous_layer_name =  generate_name(l-1,layer)
				layers[layer_name] = nonlin(np.dot(layers[previous_layer_name],synapses[previous_layer_name]))#.reshape(output_shape)
				output_layers[layer] = layers[layer_name]
				pprint("layer"+layer_name,layers[layer_name].shape)
			layers["output_layers"] = output_layers
		else :
			for layer in range(0,num_subelements):
				layer_name =  generate_name(l,layer)
				previous_layer_name =  generate_name(l-1,layer)
				layers[layer_name] = nonlin(np.dot(layers[previous_layer_name],synapses[previous_layer_name]))
			pprint("layer"+layer_name,layers[layer_name].shape)

	return layers

def create_errors_and_deltas():
	# Create errors and deltas
	errors_deltas = {}
	errors = {}
	deltas = {}
	for l in range(num_layers-1, 0, -1):
		if (l == (num_layers-1)):
			for layer in range(0,num_subelements):
				layer_name =  generate_name(l,layer)
				output_index = 0
				if (layer+1) % 2 == 0:
					output_index = 1
				errors[layer_name] = (outputData_chunked[output_index]/value_max) - layers[layer_name]
				if (j% 10000) == 0:
					prnt ("Error-"+layer_name+":"+str(np.mean(np.abs(errors[layer_name]))))
				deltas[layer_name] = errors[layer_name] * nonlin(layers[layer_name], True)
			pprint("errors"+layer_name,errors[layer_name].shape)
			pprint("delta"+layer_name,deltas[layer_name].shape)
		else:
			for layer in range(0,num_subelements):
				layer_name =  generate_name(l,layer)
				next_name = generate_name(l+1,layer)
				errors[layer_name] = deltas[next_name].dot(synapses[layer_name].T)
				reshape = remove_empty_dimensions(errors[layer_name].shape)
				errors[layer_name] = errors[layer_name].reshape(reshape)
				deltas[layer_name] = (errors[layer_name] * nonlin(layers[layer_name], True))
			pprint("error"+layer_name,errors[layer_name].shape)
			pprint("delta"+layer_name,deltas[layer_name].shape)

	errors_deltas["errors"] = errors
	errors_deltas["deltas"] = deltas

	return errors_deltas

def update_synapses():
	# Update synapses
	errors = errors_deltas["errors"]
	deltas = errors_deltas["deltas"]
	last_output = num_layers+1
	for l in range(0,num_layers-1):
		if (l == (num_layers-2)):
			for s in range(0,num_subelements):
				synapse_name = generate_name(l,s)
				next_name = generate_name(l+1,s)
				pprint("Updating synapse "+synapse_name)
				synapses[synapse_name] += layers[synapse_name].T.dot(deltas[next_name].reshape(structure_height,structure_width))
		else :
			for s in range(0,num_subelements):
				synapse_name = generate_name(l,s)
				next_name = generate_name(l+1,s)
				pprint("Updating synapse "+synapse_name,layers[synapse_name].T.shape,deltas[next_name].shape)
				synapses[synapse_name] += layers[synapse_name].T.dot(deltas[next_name])

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
inputData = np.array([
	[[[166,0,97],[108,187,5]], [[50,77,128],[60,225,153]], [[255,0,220],[18,1,200]]],
	[[[255,0,220],[18,1,200]], [[100,120,7],[180,56,6]], [[171,0,248],[166,0,97]]],
	[[[108,47,25],[60,225,153]], [[171,0,248],[166,0,97]], [[166,0,97],[108,187,5]]],
	[[[149,227,231],[180,56,6]], [[27,9,140],[108,187,5]], [[100,120,7],[180,56,6]]]
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
outputData = np.array([
	[[[255,1,87],[90,123,112]]],
	[[[103,183,212],[39,230,2]]],
	[[[9,130,0],[109,21,100]]],
	[[[39,76,200],[25,100,181]]]
])

value_max = 255
training_steps = 1
num_layers = 8
output_shape = outputData.shape
input_shape = inputData.shape
structure_width = output_shape[len(output_shape)-1]
structure_height = output_shape[0]
structure_samples = input_shape[1]
input_sample_shape = find_sample_shape()
input_sample_width = structure_width
grouping_size = max(input_sample_shape[-2],1)
num_subelements = 1
for x in input_sample_shape[1:]:
	num_subelements *=x
outputData_chunked = chunk_output()
output_layers = {}

prnt("\noutputData_chunked",outputData_chunked)
prnt("\nvalue_max",value_max,"\ntraining_steps", training_steps,"\nnum_layers",num_layers)
prnt("\ninput_shape",input_shape,"\noutput_shape",output_shape)
prnt("\nstructure_width",structure_width,"\nstructure_height",structure_height,"\nstructure_samples",structure_samples)
prnt("\ninput_sample_shape",input_sample_shape,"\ninput_sample_width",input_sample_width)
prnt("\nnum_subelements",num_subelements)

np.random.seed(1);

# Kick out if the input data is not suitable for the output
if (len(output_shape) != len(input_shape)):
	sys.exit("inputData's shape must be of the same length as the outputData's shape")

# Create synapses based on the number of layers and the dimenion of the input and output
synapses = create_synapses()

for j in range(training_steps):
	layers = create_layers()
	errors_deltas = create_errors_and_deltas()
	update_synapses()

prnt("\noutputData_chunked",outputData_chunked)
prnt("\nvalue_max",value_max,"\ntraining_steps", training_steps,"\nnum_layers",num_layers)
prnt("\ninput_shape",input_shape,"\noutput_shape",output_shape)
prnt("\nstructure_width",structure_width,"\nstructure_height",structure_height,"\nstructure_samples",structure_samples)
prnt("\ninput_sample_shape",input_sample_shape,"\ninput_sample_width",input_sample_width)
prnt("\nnum_subelements",num_subelements)
prnt("\nDesired Output:",outputData)
prnt("\noutput_layers:", output_layers)
generate_output_layer()
#prnt("\nOutput after training:", generate_output(output_layers))
