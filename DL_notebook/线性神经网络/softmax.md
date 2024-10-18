### 分类与回归
回归估计一个连续值
分类预测一个离散类别
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921215607.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921215632.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921221503.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921221536.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921221614.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921221809.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240921221815.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240922182816.png)
## 3.4.5. 小批量样本的矢量化[](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression.html#subsec-softmax-vectorization "Permalink to this heading")

为了提高计算效率并且充分利用GPU，我们通常会对小批量样本的数据执行矢量计算。 假设我们读取了一个批量的样本X， 其中特征维度（输入数量）为d，批量大小为n。 此外，假设我们在输出中有q个类别。 那么小批量样本的特征为X∈Rn×d， 权重为W∈Rd×q， 偏置为b∈R1×q。 softmax回归的矢量计算表达式为：![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240922183127.png)
相对于一次处理一个样本， 小批量样本的矢量化加快了和X和W的矩阵-向量乘法。 由于X中的每一行代表一个数据样本， 那么softmax运算可以_按行_（rowwise）执行： 对于O的每一行，我们先对所有项进行幂运算，然后通过求和对它们进行标准化。 XW+b的求和会使用广播机制， 小批量的未规范化预测O和输出概率Y^ 都是形状为n×q的矩阵。

### 3.4.6.3. 交叉熵损失[](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression.html#id10 "Permalink to this heading")

现在让我们考虑整个结果分布的情况，即观察到的不仅仅是一个结果。 对于标签y，我们可以使用与以前相同的表示形式。 唯一的区别是，我们现在用一个概率向量表示，如(0.1,0.2,0.7)， 而不是仅包含二元项的向量(0,0,1)。 我们使用 [(3.4.8)](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression.html#equation-eq-l-cross-entropy)来定义损失l， 它是所有标签分布的预期损失值。 此损失称为_交叉熵损失_（cross-entropy loss），它是分类问题最常用的损失之一。 本节我们将通过介绍信息论基础来理解交叉熵损失。
