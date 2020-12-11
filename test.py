import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sys

import matplotlib.pyplot as plt

obj = pd.Series([4,7,-5,3], index=['d','b','a','c'])
#print(obj[obj>0])
obj = np.exp(obj)

#reindex
obj = obj.reindex(['a','b','c','d','e'])
#under e is NaN

sdata = {'Ohio':3000, 'Texas': 7100, 'Cali': 2000}
obj2 = pd.Series(sdata)


states = ['Ohio','Cali','Dekota']
obj3 = pd.Series(sdata,index=states)

obj4 = obj2+obj3
obj4.name = 'population'
obj4.index.name = 'state'
#print(obj4)

data2 = {'state':['Ohio','Cali','Cali','Nvada'], 'year':[2000,2001,2000,2002],'pop':[1.5,1.7,2.3,3.2]}
#frame = pd.DataFrame(data2)
frame = pd.DataFrame(data2,columns=['year','state','pop','debt'],index=['one','two','three','four'])
#print(frame)

#frame['debt'] = np.arange(4.)
val = pd.Series([-1.2,-1.5,-1.7],index=['one','two','four'])
frame['debt'] = val
#print(frame)

frame['western'] = frame.state=='Cali'
#print(frame)

del frame['western']
#print(frame.columns)

pop = {'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.3,2001:1.7,2002:3.6},'Cali':{2000:3.6,2001:4.3}}
frame5 = pd.DataFrame(pop)
#print(frame5.T)

frame6=pd.DataFrame(pop,index=[2001,2002,2003])
#print(frame6)

labels = pd.Index(np.arange(3))
test = pd.Series([1.3,-2.5,0],index=labels)

test2 = pd.Series(['blue','purple','yellow'],index=[0,2,4])
test2 = test2.reindex(range(6),method='ffill')
#print(test2)

frame_2dim = pd.DataFrame(np.arange(9).reshape((3,3)),index=['a','b','c'],columns=['Ohio','Texas','Cali'])
#print(frame_2dim)
frame_2dim=frame_2dim.reindex(['a','b','c','d'])
#print(frame_2dim)

new_states = ['Texas','Utah','Cali']
frame_2dim = frame_2dim.reindex(columns=states)
#print(frame_2dim)

frame_2dim = frame_2dim.drop(['a','d'])
#print(frame_2dim)
frame_2dim = frame_2dim.drop('Dekota',axis = 1)
#print(frame_2dim)

#slicing
obj10 = pd.Series(np.arange(4.),index=['a','b','c','d'])
#print(obj10[2:4])
#print(obj10[['b','a','d']])
#print(obj10[obj10<2])
#print(obj10['b':'c'])
#this is inclusive

data = pd.DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
#print(data)

#get first two rows
data[:2]

#get the rows where the thsird column is greater than 5
data[data['three']>5]

#get a table of bools whether the input is <5
data<5

#set data values < 5 to 0
data[data<5] = 0

#locate the colorado row, and 2nd and 3rd column
data.loc['Colorado',['two','three']]

#locate the entire 3rd row, index start at 0
data.iloc[2]

#locate 3rd row, take in order 4th, 1st and 2nd column
data.iloc[2,[3,0,1]]

#slice matrix to rows up to Utah and take the 'two' row
data.loc[:'Utah','two']

#slice to take only rows where data.three > 5, and only the first 3 colum
data.iloc[:,:3][data.three>5]


t_data = np.arange(10)
print(plt.plot(data))



