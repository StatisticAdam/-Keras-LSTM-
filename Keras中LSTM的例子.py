from keras.layers import LSTM,Embedding,SimpleRNN,Dense
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.utils import pad_sequences

max_features=10000 #特征单词的个数
maxlen=500 #在这么多单词之后截断文本（这些单词都属于前max_features个最常见的单词）

model=Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)
input_train=pad_sequences(input_train,maxlen=maxlen) #数据的填充，使其成为相同的长度
input_test=pad_sequences(input_test,maxlen=maxlen)

history=model.fit(input_train,y_train,epochs=3,batch_size=128,validation_split=0.2)