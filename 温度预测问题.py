import os
data_dir='jena_climate_2009_2016'
fname=os.path.join('jena_climate_2009_2016.csv')
f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')#按照指定的分割符进行分割

header=lines[0].split(',')

lines=lines[1:]
print(len(header)) #15
print(len(lines))  #420451

import numpy as np
float_data=np.zeros((len(lines),len(header)-1))
#print(float_data.shape) #(420451,14)

for i ,line in enumerate(lines):
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values
#print(float_data.shape) #(420451,14)

from matplotlib import pyplot as plt
temp=float_data[:,1]#温度，单位摄氏度
plt.plot(range(1440),temp[:1440])
plt.show()


#data=0 浮点数数据组成的原始数组
lookback=720#输入数据应该包括过去多少个时间步
steps=6#数据采样的周期
dealy=144#目标应该在未来多少个时间步之后

#数据标准化
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std

#生成时间序列样本及其目标的生成器
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):#迭代器，用于生成样本和目标数据
    if max_index is  None:
        max_index=len(data)-delay-1#如果没有指定最大索引，则将其设置为数据长度减去延迟1天
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)

        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(row[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[row[j]+delay][1]
        yield samples,targets #无限生成样本和目标数据

#准备训练生成器、验证生成器、测试生成器
lookback=1440#输入数据应该包括过去多少个时间步
step=6#数据采样的周期
delay=144#目标应该在未来多少个时间步之后
batch_size=128

train_gen=generator(float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step,batch_size=batch_size)
val_gen=generator(float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,shuffle=True,step=step,batch_size=batch_size)
test_gen=generator(float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,shuffle=True,step=step,batch_size=batch_size)
val_steps=(300000-200001-lookback)//batch_size #为了查看整个验证集，需要从val_gen中抽取多少次
test_steps=(len(float_data)-300001-lookback)//batch_size#为了查看整个测试集，需要从test_gen中抽取多少次

#训练并评估一个基于GRU的模型
# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model=Sequential()
# model.add(layers.GRU(32,input_shape=(None,float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(),loss='mae')
# history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

#训练并评估一个使用dropout正则化的基于GRU的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model=Sequential()
model.add(layers.GRU(32, dropout=0.2,recurrent_dropout=0.2,input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

model=Sequential()
model.add(layers.GRU(32, dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64,activation='relu', dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)

model=Sequential()
model.add(layers.Bidirectional(layers.GRU(32),input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)
