![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003180251.png)

Encoder 的 状态h中积累输入的信息，最后一个状态hm中积累了所有词向量x的信息。encoder输出最后一个向量hm，把之前的向量状态全部扔掉。
decoder Rn的初始状态s0 = encoder rn的最后一个状态hm，hm包含了输入英语状态的信息，通过hm，decoder就知道了这句英语。
decoder就像是一个文本收集器一样，逐字生成一句目标语言。
可惜seq2seq模型有一个明显的缺陷，要是输入的句子很长，那么encoder会记不住完整的句子。encoder最后一个状态可能会漏掉一些信息。
加入有个别词被忘记了，那么decoder就不能得知完整的句子，也就不能给出正确的翻译。

如果拿seq2seq模型做机器翻译，结果如下
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003180313.png)
，用了attention后，结果如下：
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003180348.png)
使用了attention后，每次更新状态的时候，都会再看一遍encoder所有状态，这样就不会遗忘。
attention还会告诉decoder应该关注decoder哪个状态。
再encoder已经结束工作后，attention和decoder同步开始工作。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003180825.png)

这里需要计算s0与每一个h向量的相关性。encoder有m的状态，所以一共算m个α。从1到m，αi都是0-1之间的实数，所有α加起来的和为1。

计算α（hi与s0的相关性）
第一种：把hi与s0做concat，得到更高的向量，然后求矩阵w与这个向量的乘积，在使用tanh作为激活函数，把每一个元素都压到-1与+1之间，最后计算向量v与刚算出来的向量的内积。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003181134.png)
第二种：更多的被transformer所使用。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003181252.png)

利用这些权重α，我们可以对这m个状态向量h做加权平均，计算α1h1 + α2h2 +....+αmhm = c0
将c0记作context vector，每一个context vector都会对应一个decoder状态s，
c0对应状态s0，decoder读入向量x1‘（输入），然后需要把状态更新为s1。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003182609.png)

如果不用attention，更新方法为：
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003181913.png)
simpleRNN在更新状态的时候只需要知道输入x1与旧状态s0即可。他并不会去看encoder 的状态。
如果用了attention，更新方法为：
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003182351.png)
于是RNN遗忘的问题就解决了。
接下来要计算c1，αi是hi与s1的相关性。
把decoder状态s1，与encoder所有m个状态对比，计算出m个权重，记作α1到αm，注意：就算上一次计算c0的时候算出了所有m个权重α，但是我们现在不能用那些权重。必须要重新计算α，上一轮算的是h与s0的相关性，这一轮要的是h与s1的相关性。
有了权重α就可以计算新的context vector c1，c1是encoder m个状态向量h1到hm的加权平均。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003184123.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003184136.png)
依次往后类推，一直计算出decoder内的所有s向量。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003184230.png)
总结：
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003184457.png)
通过巨大的计算来确保准确度。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003184853.png)

粗的线表示α很大，细的线表示α很小，两个状态的相关性可以通过线条的粗细反应。每当decoder想要生成一个状态的时候，都会看一遍encoder所有的状态，这些权重α告诉decoder应该关注什么地方，这就是attention名字的由来，当decoder需要计算某个状态的时候，权重α告诉decoder应该关注哪一个词缀，这样帮助decoder产生正确的状态，从而生成正确的目标单词。

summary
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003185028.png)
