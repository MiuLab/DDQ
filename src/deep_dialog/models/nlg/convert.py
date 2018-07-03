import cPickle
model=cPickle.load(open('lstm_tanh_relu_[1468202263.38]_2_0.610.p'))
cPickle.dump(model,open('model.bin.nlg','wb'))