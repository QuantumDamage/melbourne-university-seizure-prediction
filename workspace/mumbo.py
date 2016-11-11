import glob
import scipy.io as sio
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D

seed = 7
numpy.random.seed(seed)

how_many_files_i_try_to_use = 2
X_list = []
Y_list = []

train_filenames = glob.glob("../input/train_*/*")

train_filenames = train_filenames[:how_many_files_i_try_to_use]

for train_filename in train_filenames:
    X_list.append(sio.loadmat(train_filename)["dataStruct"][0][0][0])
    Y_list.append(int(train_filename[-5:-4]))

X = numpy.array(X_list)
Y = numpy.array(Y_list)

model = Sequential()
#model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Convolution1D(12, border_mode='same', input_shape=(240000,16)))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=150, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
