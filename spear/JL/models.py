import torch.nn as nn

class LogisticRegression(nn.Module):
	'''
		Class for Logistic Regression, used in Joint learning class/Algorithm

	Args:
		input_size: number of features
		output_size: number of classes
	'''
	def __init__(self, input_size, output_size):
		'''
		'''
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		'''
		'''
		return self.linear(x)

class DeepNet(nn.Module):
	'''
		Class for Deep neural network, used in Joint learning class/Algorithm
		
	Args:
		input_size: number of features
		hidden_size: number of nodes in each of the two hidden layers
		output_size: number of classes
	'''
	def __init__(self, input_size, hidden_size, output_size):
		'''
		'''
		super(DeepNet, self).__init__()
		self.linear_1 = nn.Linear(input_size, hidden_size)
		self.linear_2 = nn.Linear(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		'''
		'''
		out = nn.functional.relu(self.linear_1(x))
		out = nn.functional.relu(self.linear_2(out))
		return self.out(out)