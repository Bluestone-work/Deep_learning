seq2seq只会用输入序列的最后一个词作为状态，他看不到前面的词，目的是，我想让翻译对应的词的时候去让注意力关注到原句子中对应的部分，这个就是说把注意力机制放在里面的一个动机。![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003165543.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003165552.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003165556.png)
在做预测的时候，contest的信息不应该用的是.的，而应该是用hello的+bos的embedding。
接着在算Bonjour的输出时候，传入的应该是world对应的contest和Bonjour对应的embedding。
### key value
意思就是假设你的英语句子长为3的话，你就会拥有3个key-value pair，这个key和value是等价的，他的意思就是第i个词他的RNN的输出。

### query
解码器RNN对上一个词的输出是qurey，比如说，上一次我预测的值是hello的话，接下来我要去翻译下一个词，用这个词去查他应该去找到hello那个英语表示的词，然后把附近的词给圈出来。（预测出hello，然后qurey放到attention中就是寻找hello附近的词）
再打个比方：source是你好世界，target是hello world，现在你预测得到hello，根据hello去预测world，所以query就是hello。要预测的词的上一个词的向量跟encoder的隐藏项做attention交互得到要预测的值。

## 加入注意力（可以与前面不使用注意力机制作对比[[序列到序列学习]]）
![](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003171509.png)

	1.编码器对每次词的输出作为key和value
	2.解码器RNN对上一个词的输出是qurey（预测出你好，将你好作为query，进而放入attention中找到hello附近的词（k-v），然后将这个k-v作文上下文和输入的embedding一起放入RNN）
	3.注意力的输出和下一个词的词嵌入合并进入RNN
与之前的seq2seq不同的是解码器拼接上下文要用注意力机制从编码器多个时间步最后的输出中做选择。之前的上一个RNN里面最后那一个词的输出，现在我把所有的词拿出来，做一个weight，做一个加权的平均，根据你放的词不一样，可能一开始我用前面的一些输出，越到后面我用越后面的词的输出。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241003173119.png)
[[【论文解读】Bahdanau Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/272662664)]()
