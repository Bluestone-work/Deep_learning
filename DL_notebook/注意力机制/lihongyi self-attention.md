![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004160159.png)
word embedding会给每个词汇一个向量，并且这些向量是长度不一的。![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004160757.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004161857.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004162041.png)
语音辨识，指不定是几个类别。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004162419.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004162600.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004162627.png)
	![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004171007.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004171147.png)

要求出b1我们要先找出a里面与a1相关的向量，每一个向量和a1相关的成都用α表示。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004171255.png)
这时候需要一个计算attention的模组，比较常见的做法有Dot-product和Additive。Dot-product是最常用的方法，也是用到transformer的方法。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004171454.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004171527.png)

这是就要把a1和a2，a3，a4都计算相关性。最后也要对自己求相关性，将求出来的值放入softmax中算出a‘。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004173041.png)
根据α’，我们要从这之中抽出相关性资讯。判断出哪些向量是和a1最相关的。
b1到bn不是顺序计算出来的，而是并行一起计算出来的
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004182941.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004183138.png)
## 矩阵乘法角度，讲述self-attention。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004183703.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004184927.png)
softmax不是唯一的选项，可以用relu之类的激活函数，效果是差不多的。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185142.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185228.png)
## 进阶，Multi-head Self-attention
我们不能只有一个Q，我们应该要有多个Q，不同的Q负责不同种类的相关性。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185527.png)
比如说，我们认为这个类别里有两个不同的相关性。，每一种相关性都之和他自己的类别做运算。![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185606.png)
## 位置资讯
对self-attention而言，这些位置是一样的，q1到q4和q1到q2的距离都是一样的，但是这样子设计是有问题的。有时候位置是一个很重要的概念。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185856.png)
早期的时候每一个位置都有一个独特的e，这些位置是人设定的。
如今是用sin和cos函数产生的。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004185944.png)
总之有各种各样的方法产生position encoding
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004190051.png)
我们在做语音识别的时候现在都是用一种叫Truncated self-attention的方法。认为设定一个范围，不用看整个部分。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004190244.png)
## self attention用在影像上
到目前位置，self-attention适用的范围，输入是一排向量的时候，输入是一个vector set的时候，他适合使用。
cnn，一张图片看作是一个很长的向量，也可以把它看作是一个vector set。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004190656.png)
## self-attention vs CNN

![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004231413.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004231429.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004231622.png)

## self-attention vs RNN
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004231852.png)

## self-attention for graph
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004232102.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241004232123.png)

