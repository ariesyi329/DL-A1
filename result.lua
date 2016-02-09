require('torch')
require('nn')
require('xlua')

data_path = 'mnist.t7'
model_path = "results"

model_file = paths.concat(model_path, 'model_128x128x256_epoch10.net')
model = torch.load(model_file)

print "==> loading model..."

train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

trsize = 60000
tesize = 10000

loaded = torch.load(train_file, 'ascii')
trainData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

trainData.data = trainData.data:float()
testData.data = testData.data:float()

print '==> preprocessing data: normalize globally'
mean = trainData.data[{ {},1,{},{} }]:mean()
std = trainData.data[{ {},1,{},{} }]:std()
trainData.data[{ {},1,{},{} }]:add(-mean)
trainData.data[{ {},1,{},{} }]:div(std)

-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)

trainMean = trainData.data[{ {},1 }]:mean()
trainStd = trainData.data[{ {},1 }]:std()

testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()

print('training data mean: ' .. trainMean)
print('training data standard deviation: ' .. trainStd)

print('test data mean: ' .. testMean)
print('test data standard deviation: ' .. testStd)


model:evaluate()
preds = {}

for t = 1,testData.size() do

	xlua.progress(t, testData.size())

	local input = testData.data[t]
	input = input:double()

	local pred = model:forward(input)

	local label = torch.LongTensor()
	local _max = torch.FloatTensor()
	_max:max(label, pred:float(), 1)
	preds[t] = label[1]

end

print('==> saving predictions')
file = io.open('predictions.csv', 'w')
file:write('Id,Prediction\n')
for i, p in ipairs(preds) do
   file:write(i..','..p..'\n')
end
file:close()


