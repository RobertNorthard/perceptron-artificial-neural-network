

unit_step = lambda x : 1 if x > 0.5 else 0

def activation(input_vector, weights):

	activation_value = 0.0

	for i in xrange(0,len(input_vector)):
		activation_value += input_vector[i] * weights[i]

	# add bias to control linear classifier horizontal shift
	activation_value += 1 * weights[len(weights) - 1]

	return activation_value

def activation_function(input_vector, weights):
	return unit_step(
		activation(input_vector, weights))

def train(training_set, learning_rate, epoch):

	weights = [0.0] * len(training_set[0])

	for epoch_count in range(epoch):
		rmse = 0.0

		for vector in training_set:
			expected = vector[len(vector) - 1]
			actual = activation_function(vector, weights)
			delta = expected - actual
			rmse += delta**2

			for i in range(len(weights) - 1 ):
				weights[i] += learning_rate * delta * vector[i]

			weights[len(weights) - 1] += learning_rate * delta

		print "> epoch=%.f RMSE=%.2f weights=%s" % (epoch_count,rmse, weights)

	return weights

training_set = [
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,1]
]

learning_rate = 0.2
epoch = 5
 
print "Learning rate=%s" % (learning_rate)
print "Training Epoch=%s" % (epoch)
print "Training Data=%s" % (training_set)

print
print "Training network."

weights = train(training_set, learning_rate, epoch)
print "Network trained"
print 

print "Testing network has learnt Logical AND"
for vector in training_set:
	print "> input vector=%s expected result=%s result=%s" % (
		vector[:-1],
		vector[len(vector) - 1],
		activation_function(vector[:-1], weights))

