-- This is a script to sample dataset.

require 'torch'

-- load dataset
train_file = 'mnist.t7/train_32x32.t7'
trsize = 60000

print '==> loading dataset'

loaded = torch.load(train_file, 'ascii')
train_data = {
	data = loaded.data,
	labels = loaded.labels,
	size = function() return trsize end
}

function splitDataset(data, ratio)
	d = data.data
	l = data.labels
	shuffle = torch.randperm(data:size(1))
	numTrain = data:size(1)*ratio
	numVal = data:size(1)*(1-ratio)
	trainData = torch.Tensor(numTrain, d:size(2), d:size(3), d:size(4))
	valData = torch.Tensor(numVal, d:size(2), d:size(3), d:size(4))
	trainLabels = torch.Tensor(numTrain)
	valLabels = torch.Tensor(numVal)
	for i = 1, numTrain do
		trainData[i] = d[shuffle[i]]
		trainLabels[i] = l[shuffle[i]]
	end
	for i = 1, numVal do
		j = i+numTrain
		valData[i] = d[shuffle[j]]
		valLabels[i] = l[shuffle[j]]
	end
	return trainData, trainLabels, valData, valLabels
end

trainData, trainLabels, valData, valLabels = splitDataset(train_data, 0.7)
train = {
	data = trainData,
	labels = trainLabels,
	size = trainData:size(1)
}
validation = {
	data = valData,
	labels = valLabels,
	size = valData:size(1)
}
torch.save('data/train.t7', train, 'ascii')
torch.save('data/validate.t7', validation, 'ascii')
