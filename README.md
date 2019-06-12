# NNregression
普通单层神经网络回归模型函数（训练+使用 ）


## 先使用train.py
单隐藏层训练模型函数
----------------
训练函数：  train(x, y, n, epoch_max, lr, mod=F.relu)  
使用方法：  
x: 特征属性,例：   
x = [[1,2,3],[2,3,4]]  ([x1,x2]), x1=[...], x2=[...]  
y: label，标签，目标属性  
y = [1,2,3]     y = [...]  
  
x1   x2   y  
1    2    1  
2    3    2  
3    4    3  
  
n 单隐藏层的结点数  
epoch_max: 最大训练次数  
lr: 学习率 学习率低，学的比较慢 参数一种  
mod: 激励函数，默认为 F.relu， 若替换则可使用F.tanh, F.sigmoid 等  
train(x, y, 10, 100, 0.01) ->   
返回：生成记录属性平均值和标准差的csv  和 神经网络信息 csv 和 神经网络模型参数  
  
## 再使用use.py
使用模型  
----------------
使用函数： pred(x)  
使用方法：  
x: 特征属性,例：    
x = [1,2]     ([x1, x2]) float/int  
返回：预测值 float y  

# 用法
### 1.将NNtrain.py和NNuse.py放在当前目录下
### 2.将数据调整成规定列表形式
### 3.导入train和pred函数
```python
from NNtrain import train
from NNuse import pred
```
### 4.使用train函数，例：train(x, y, 10, 100, 0.01)，会生成'mean_div.csv','NN.txt',和'net.pt'
```python
train(x, y, n, epoch_max, lr, mod=F.relu)  
```
### 5.使用pred函数，例：pred(x),得到float类型的回归值
```python
pred(x)  
# 此处x是需要得出预测的值列表
```