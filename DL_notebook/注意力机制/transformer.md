
与前面的![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005161832.png)架构不同的地方在于，我们把其中的RNN层换成了transformer。![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005161912.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005161928.png)

## 多头注意力Multi-head
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005163647.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005163652.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005163659.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005164011.png)
f有两个版本，前文说过。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005164022.png)
## 有掩码的多头注意力
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005164130.png)
decoder时，他看不到后面的信息。
这里应该是同时用了两种掩码，一种是前面说过的padding mask还有一种是sequence mask，老师这里讲的是第二种

## 基于位置的的前馈网络
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005164249.png)
全连接
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005164258.png)
b batch_size,n 序列长度,d 向量长度 = 即每个输入的特征数量 = dimention

如果这个网络处理的是自然语言处理（NLP）任务，输入通常是经过嵌入（embedding）后的数据，那么“d”可能代表的是嵌入向量的维度。而词汇表大小（vocab_size）通常只会在嵌入层或输出层中使用，不直接与这个公式中的“d”相关。

因此，在这个网络结构中，**“d” 更可能是特征向量的维度，而不是词汇表的大小**。

### 1. **词汇表大小（vocab_size）**

- **定义**：词汇表大小是一个整数，表示模型所能处理的词汇总量。每个唯一的单词或子词都被分配了一个唯一的索引。
- **作用**：在处理自然语言时，文本数据首先需要通过**嵌入层（embedding layer）**进行转换，模型需要知道每个词汇的唯一标识。这就是为什么词汇表大小对于嵌入层非常重要。

### 2. **嵌入层（Embedding Layer）**

- **输入**：模型的输入是形如 `(b, n)` 的二维张量（Tensor），其中 `b` 是批大小（batch size），`n` 是序列的长度（通常是句子的单词数）。每个元素是词汇表中的一个索引（整数），范围是 `[0, vocab_size)`。
- **输出**：通过嵌入层后，词汇索引会被转换为相应的**嵌入向量**。输出的张量形状变为 `(b, n, d)`，其中 `d` 是嵌入向量的维度（embedding size）。这个嵌入向量就是每个单词的低维表示。
- **作用**：嵌入层的作用是将每个词汇索引映射到一个 `d` 维的实数向量空间中，使得单词之间的关系在这个向量空间中能被更好地表示。

### 3. **嵌入向量维度（d）**

- **定义**：嵌入向量的维度 `d` 决定了每个词汇在模型内部的表示方式。这个 `d` 值通常是可以调节的超参数，常见的值包括 128、256、512 等。
- **作用**：`d` 的大小决定了每个词的表示精度。维度越高，理论上可以学习到越复杂的词汇关系，但同时也增加了计算和存储的成本。

### 4. **前馈网络（Feedforward Network）**

- **输入和输出形状的变化**：从你提供的图片来看，前馈网络的输入形状是 `(b, n, d)`，代表一个包含 `b` 个句子（或其他序列），每个句子长度为 `n`，每个单词或位置的表示维度为 `d` 的输入。
    
    - 将 `(b, n, d)` 转换为 `(b * n, d)`，这是将句子的批次（batch）展平成一个单一维度，用于全连接层处理。
    - 经过两个全连接层后，维度会保持不变（仍然是 `(b * n, d)`），这意味着模型对每个位置的表示进行了进一步的处理。
    - 之后再将形状重新调整为 `(b, n, d)`，以匹配原始的输入形状。
### 5. **嵌入向量（Embeddings）**

- **定义**：嵌入向量是一种将词汇映射到低维连续向量空间中的方法，每个词汇都会被表示为一个维度为 `d` 的实数向量。这个向量空间是通过模型学习得到的，能捕捉到词汇之间的语义相似性。
    
- **例子**： 假设嵌入向量的维度 `d=3`，单词 "apple" 的嵌入向量可能是 `[0.1, 0.7, 0.4]`，而单词 "banana" 的嵌入向量可能是 `[0.2, 0.6, 0.5]`，这种表示方式能够捕捉到词汇之间的语义相似性和关系。
    
- **生成方式**：
    
    - 嵌入向量**不是**通过 one-hot 编码生成的，而是通过**嵌入层（embedding layer）**或**预训练的嵌入矩阵**生成的。在嵌入层中，每个词汇对应一个 `d` 维度的嵌入向量，这些向量在模型训练过程中逐步优化，学习到更好的语义表示。
    - 这种方式的维度（`d`）通常远远小于词汇表的大小（`vocab_size`），例如 `d=300`，这大大减少了存储和计算的开销，并且通过梯度下降等方法来更新这些嵌入向量，使得词汇之间的语义关系能够被更好地捕捉。
- **优势**：
    
    - 嵌入向量可以捕捉到词汇的语义相似性和关系。例如，"apple" 和 "banana" 可能在向量空间中距离较近，而 "apple" 和 "dog" 之间的距离较远。
    - 嵌入向量的维度 `d` 可以被设置为一个相对较小的值（如 50、100、300等），而不会受到词汇表大小的限制。

### **从 One-hot 编码到嵌入向量的关系**

虽然嵌入向量和 one-hot 编码都是词汇的表示方式，但它们的生成过程和使用方式是不同的：

- 在某种意义上，one-hot 编码可以被视为嵌入向量生成过程中的一种中间表示。例如，在嵌入层中，模型实际上会将输入的词汇索引（可以视作是 one-hot 编码）作为查找嵌入矩阵（embedding matrix）的索引，从而得到对应的嵌入向量。
- **嵌入层的机制**：在嵌入层中，每个词汇的索引对应一个嵌入矩阵中的行，该行就是词汇的嵌入向量。因此，one-hot 编码可以被视为查找嵌入向量的“索引方式”，而不是最终用于训练和推理的表示。

### **总结嵌入向量与 One-hot 编码的区别和关系**

- **One-hot 编码**：
    - 向量长度等于词汇表大小（`vocab_size`）。
    - 只能表示词汇的唯一性，无法表示词汇之间的关系。
    - 维度高，存储和计算成本大。
- **嵌入向量**：
    - 向量长度为 `d`，`d` 是嵌入维度，通常远小于词汇表大小。
    - 嵌入向量可以表示词汇之间的语义相似性，通过学习得到。
    - 维度较小，计算和存储成本低，适合大规模语料库。

**嵌入向量的生成并不是通过 one-hot 编码直接计算的**，而是通过词汇索引（可能与 one-hot 编码有关）在嵌入矩阵中查找，或通过训练优化获得的。

### **嵌入层的训练**

在模型训练过程中，嵌入层会逐渐学习词汇之间的关系，通常通过反向传播算法，更新嵌入矩阵中的值，使得常见的、语义相似的词汇距离更近。这种学习可以是模型内部自我学习的结果（如 Transformer 中的嵌入层），也可以来自于预训练的词向量（如 Word2Vec 或 GloVe 等）。

## 层归一化ADD&Norm
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005165219.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005165227.png)
这个地方不能用batch_normalization,而要用layer_normalization,因为序列的长度是会变化的。
1. **批量归一化（Batch Normalization）**：
    
    - 对每个特征或通道里的元素进行归一化。
    - 主要在每个批次的数据（例如图中的 b 表示 batch size）中，对每个通道（d 代表 feature dimension）进行归一化处理。
    - 不适合序列长度（如 NLP 应用中序列长度可能会变化的情况），因为 Batch Norm 针对的是批量维度。
2. **层归一化（Layer Normalization）**：
    
    - 对每个样本的元素进行归一化，不依赖于批次大小。
    - 每个样本（图中 len 代表序列长度，d 代表特征维度）中的所有元素进行归一化处理，不针对 batch 维度归一化，因此更适合像 NLP 中长度可能变化的场景。
3. **ADD & Norm 机制**：
    
    - 在深度学习模型中，通常会使用 Layer Normalization，在残差连接后进行归一化（如图左下角所示的残差连接 "+" 和 Block）。
    - 在每个层结束后，通过 Layer Norm 对该层的输出进行标准化处理，以加快训练和稳定模型。

### 总结：

- 批量归一化（Batch Norm）适用于图像处理等场景，因为数据的形状通常是固定的。
- 层归一化（Layer Norm）更适合 NLP 等序列长度变化的任务，它对每个样本进行归一化，不依赖于批次的大小。

## 信息传递
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005170549.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005170602.png)

## 预测
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005170754.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005170834.png)

## 总结
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005170933.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20241005232734.png)
