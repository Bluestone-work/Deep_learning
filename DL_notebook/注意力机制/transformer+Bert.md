transformer就是一个seq2seq的model。

输入和输出的长度是不存在很大关系的
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004235722.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004235856.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004235956.png)

## 不论是语音，还是文字上都可以用得到
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005001345.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005001614.png)
要对特定的问题做特质化各式各样的模型。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005001820.png)
文法解析
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005001831.png)
目标检测
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005001912.png)
## seq2seq
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005002020.png)

## Encoder

encoder内，都是一排向量输入到一个block，再从block输出为一排向量，依次反复。每一个block并不只是一层，而是很多层。![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005111053.png)
transformer中的某一个block内做的事
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005111328.png)

![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005111450.png)
这些块不一定要这么设计也行。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005111539.png)

## Decoder

### 过程
#### step1（以begin和encoder的输出作为decoder的第一个输入）
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005113434.png)
#### step2（以第一个begin的输出向量和第二个begin作为输入）
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005113600.png)
#### step3（依次反复的重复下去）
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005113748.png)

### 内部结构
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005113925.png)
如果不看中间一块，其实encoder和decoder没有很大差别。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005114303.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005114329.png)
对decoder而言，是现有a1才有a2依次顺序的，这点和encoder是不一样的，所以我们要用masked self-attention。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005114508.png)
那么我们又怎样才能知道输出的seq的长度呢？
decoder是没法知道什么时候停下来的，所以我们要加入一个符号，段‘END’。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005115455.png)

### Non-autoregressive（NAT）
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005120201.png)

### Cross Attention
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005120323.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005124320.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005124345.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005124602.png)
不用和原始的transformer结构一样，decoder所用的只能是encoder最后一层输出的隐藏状态，也可以自己改变。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005124851.png)

## Trainning
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005155735.png)
正确的答案当作decoder的输入

## Copy Mechanism
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005160018.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005160043.png)
