import time
start_time = time.time()
import glob
import scipy.io as sio
import numpy
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D

seed = 7
numpy.random.seed(seed)

how_many_files_i_try_to_use = 32
X_list = []
Y_list = []


if len(train_filenames) != how_many_files_i_try_to_use:

    files_1 = int(round(how_many_files_i_try_to_use * ratio1to0))
    files_0 = int(how_many_files_i_try_to_use - files_1)

    train_filenames = glob.glob("../input/train_*/*")
    train_filenames_0 = glob.glob("../input/train_*/*_0.mat")
    train_filenames_1 = glob.glob("../input/train_*/*_1.mat")
    ratio1to0 = len(train_filenames_1)/len(train_filenames_0)

    train_filenames = train_filenames_0[:files_0] + train_filenames_1[:files_1]

    for train_filename in train_filenames:
        X_list.extend(numpy.squeeze(numpy.hsplit(sio.loadmat(train_filename)["dataStruct"][0][0][0],16)))
        minilist = []
        minilist.append(int(train_filename[-5:-4]))
        Y_list.extend(minilist*16)

    X = numpy.array(X_list)
    Y = numpy.array(Y_list)



model = Sequential()
model.add(Dense(24, input_dim=240000, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=150, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



#submissions:
def determine_result(filename):
    test_set=filename[0]
    data_path="../input/test_" + test_set + "/" + filename
    #print(data_path)
    data_to_test = []
    data_to_test.extend(numpy.squeeze(numpy.hsplit(sio.loadmat(data_path)["dataStruct"][0][0][0],16)))
    X_test = numpy.array(data_to_test)
    all_predicts = model.predict(X_test)
    return (int(round(sum(all_predicts)/len(all_predicts))))
    
print ("generating submissions")
print (time.time() - start_time)
print ("---------------")

submission = pd.read_csv("../input/sample_submission.csv")
#submission["Class"] = np.random.randint(0,2,size=len(submission["Class"]))
#submission["Class"] = determine_result(submission["File"])
submission["Class"] = submission['File'].apply(lambda x: determine_result(x))
print(submission.describe())
submission.to_csv("submission.csv", index=False)

print(time.time() - start_time)
