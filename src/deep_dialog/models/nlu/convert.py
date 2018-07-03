import cPickle
model=cPickle.load(open('lstm_[1468447442.91]_39_80_0.921.p'))
cPickle.dump(model,open('model.bin.nlu','wb'))