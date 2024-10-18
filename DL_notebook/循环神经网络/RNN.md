## 概念
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928143436.png)
计算损失时是比较Ot和Xt之间的关系来计算损失。但是Xt是用来更新你的ht使得他移动到下一个单元。这一点和MLP不一样，他有个时间轴。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928143422.png)
他的更新方式：假设ht是一个隐藏状态，φ就是一个激活函数，
观察Whx，他就是MLP隐藏层的weight，xt-1时一个输入。
但是他也和前一个ht-1相关，有一个单独的weight，所以ht的式子要加上一项。
拿到了隐藏状态之后，就和MLP一样可以进行输出。这时候是不需要φ的，图上多了φ。
## 使用循环神经网络的语言模型
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928143639.png)

## 困惑度
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928145736.png)
语言模型说白了就是一个分类模型，我的输出就是判断下一个词，假设我的字典大小是m，就是做一个m类的分类问题，因此我们可以用交叉熵。
n就是对一个序列做n词分类。困惑度的大小可以理解为下一个词可能是多少个词。

## 梯度剪裁
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928150258.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928150304.png)
g是所有层上的梯度，假设我的g的模长超过了θ，那么就把他降回θ。
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928150651.png)
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928150916.png)
## 总结
![image.png](https://cdn.jsdelivr.net/gh/Bluestone-work/image/image/20240928151010.png)
如果t很长的话，就会发生数值稳定性的问题，所以说RNN一般都需要数值剪裁。

`.detach_()` 是 PyTorch 中的一个方法，作用是将一个张量（或隐藏状态）从当前的计算图中分离出来，防止在后续的计算中继续跟踪它的梯度。这在训练循环中非常常见，尤其是处理 RNN、LSTM 或 GRU 这类有时间依赖性的模型时。具体作用如下：

```
#@save

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):

    """训练网络一个迭代周期（定义见第8章）"""

    state, timer = None, d2l.Timer()

    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量

    for X, Y in train_iter:

        if state is None or use_random_iter:#时序上不连续，上一个state不能用在下一个

            # 在第一次迭代或使用随机抽样时初始化state

            state = net.begin_state(batch_size=X.shape[0], device=device)

        else:

            if isinstance(net, nn.Module) and not isinstance(state, tuple):

                # state对于nn.GRU是个张量

                state.detach_()#把上一个小批量的样本传过来。

            else:

                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量

                for s in state:

                    s.detach_()

        y = Y.T.reshape(-1)

        X, y = X.to(device), y.to(device)

        y_hat, state = net(X, state)

        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):

            updater.zero_grad()

            l.backward()

            grad_clipping(net, 1)

            updater.step()

        else:

            l.backward()

            grad_clipping(net, 1)

            # 因为已经调用了mean函数

            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())

    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```
### 代码主要流程：

1. **遍历训练数据集（`train_iter`）**：
    
    - `train_iter` 是一个可迭代对象，每次迭代会返回一对输入 (`X`) 和目标 (`Y`) 的小批量数据。这是模型进行训练的数据。
2. **初始化隐藏状态（`state = net.begin_state(...)`）**：
    
    - `state` 代表RNN、LSTM或GRU的隐藏状态（hidden state）。隐藏状态是序列模型捕捉时间依赖关系的重要部分。
    - `net.begin_state(...)` 是一个方法，用于初始化隐藏状态，通常初始化为全零张量或根据具体模型的需要进行初始化。
3. **判断是否需要重新初始化隐藏状态（`if state is None or use_random_iter`）**：
    
    - 如果 `state` 为 `None` 或者 `use_random_iter` 为 `True`，说明我们需要重新初始化隐藏状态。这种情况通常在两种情况下发生：
        - **`state is None`**：通常是第一次迭代时，隐藏状态尚未初始化。
        - **`use_random_iter`**：表示使用随机抽样，这种情况下小批量数据在时间序列上不连续，因此无法传递上一个批次的隐藏状态，需要重新初始化。
    - 当数据是随机抽样时（即打乱顺序的小批量数据），每个批次的隐藏状态都是独立的，不能传递隐藏状态，因为这些数据之间没有时间上的关联。
4. **处理连续的小批量数据（`else` 分支）**：
    
    - 如果 `state` 已经有值，表示这是一个时间上连续的数据批次，隐藏状态需要传递到下一个批次。
    - **`if isinstance(net, nn.Module) and not isinstance(state, tuple)`**：
        - 如果网络 `net` 是一个 `nn.Module`（比如 PyTorch 中的模型），并且 `state` 不是一个元组，说明使用的是 `GRU` 之类的模型，它的隐藏状态是一个张量。
        - **`state.detach_()`**：将上一个批次的隐藏状态从计算图中分离，避免梯度回传时计算图不断增长导致内存溢出。
    - **`else`**：
        - 如果 `state` 是元组（如 LSTM 的隐藏状态是 `(h, c)`，分别代表隐藏状态和细胞状态），则对元组中的每个元素都执行 `detach_()`，同样是为了分离梯度，避免计算图爆炸。
5. **对目标 `Y` 进行变换（`y = Y.T.reshape(-1)`）**：
    
    - 这里对目标 `Y` 进行了转置（`Y.T`），然后将其形状拉平成一维向量（`reshape(-1)`）。这通常是为了将目标数据的形状与模型的输出对齐，因为 RNN 模型的输出通常是一系列的时间步长对应的预测结果。

### 总结：

- **训练时序数据**：
    - 如果数据是连续的（即非随机抽样），隐藏状态会在小批量数据之间传递，以便捕捉长时间的依赖关系。
    - 当使用随机抽样时，每个批次的数据都是独立的，因此需要重新初始化隐藏状态，不能传递过去。
- **高效管理隐藏状态**：
    - `detach_()` 操作用于分离隐藏状态的梯度，只针对当前批次计算梯度，避免计算图逐渐变得过于复杂，导致内存问题。
### `.detach_()` 的作用：

1. **防止计算图扩展**：
    
    - 在深度学习中，每一次前向传播都会构建一个计算图，用来存储所有操作的历史记录，以便在反向传播时计算梯度。
    - 如果不使用 `.detach_()`，每次前向传播的计算图都会与上一个批次的计算图连接在一起，这会导致计算图变得非常大，耗费大量内存，并且在反向传播时计算代价会非常高。
    - `.detach_()` 的作用是将张量从计算图中分离出来，切断它与之前的计算历史的关联，从而防止计算图不断增长。
2. **保留张量的值，但不计算梯度**：
    
    - 使用 `.detach_()` 后，张量本身的值会保留下来，但 PyTorch 不会再跟踪其梯度。也就是说，虽然这个张量会继续参与后续的计算，但这些计算不会影响之前的梯度记录。
    - 这非常重要，特别是在处理 RNN 或 LSTM 这种跨时间步长传递隐藏状态的模型中，我们希望隐藏状态保留值并传递到下一个时间步，但不希望计算图无限制地扩展下去。
3. **使用场景**：
    
    - 在序列模型（如 RNN、LSTM、GRU）中，我们经常会在每个时间步或每个批次处理完数据后调用 `.detach_()`，以便将隐藏状态与之前的计算分离出来。这允许我们在新批次上继续训练，而不会影响前面批次的计算图。
    - 例如，当处理一个长序列时，隐藏状态会在每个批次之间传递。每次传递之前，使用 `.detach_()` 切断对前一个批次的依赖，可以防止内存消耗不断增加，同时加速训练。

```
y = Y.T.reshape(-1)

        X, y = X.to(device), y.to(device)

        y_hat, state = net(X, state)

        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):

            updater.zero_grad()

            l.backward()

            grad_clipping(net, 1)

            updater.step()

        else:

            l.backward()

            grad_clipping(net, 1)

            # 因为已经调用了mean函数

            updater(batch_size=1)

        metric.add(l * y.numel(), y.numel())
```

这段代码是一个典型的神经网络训练循环中的一部分，尤其是在训练循环中涉及到递归神经网络（RNN）、LSTM 或 GRU 时。它展示了如何进行前向传播、计算损失、反向传播和梯度裁剪。以下是对代码的中文解释和逐步分析：

### 代码解析：

1. **`y = Y.T.reshape(-1)`**：
    
    - 这行代码对目标标签 `Y` 进行处理。首先，将 `Y` 进行转置（`Y.T`），然后使用 `reshape(-1)` 将其展平为一维向量。
    - 这个操作是为了将目标标签整理成模型输出可以比较的形状，通常在处理序列数据时需要进行这种操作。
2. **`X, y = X.to(device), y.to(device)`**：
    
    - 这行代码将输入 `X` 和目标 `y` 移动到指定的设备上（通常是 `GPU` 或 `CPU`），以确保模型和数据位于相同的设备上进行计算。
3. **`y_hat, state = net(X, state)`**：
    
    - `net(X, state)` 进行前向传播，将输入 `X` 和隐藏状态 `state` 传递给模型，模型输出预测值 `y_hat` 和更新后的隐藏状态 `state`。
    - 对于递归神经网络模型，隐藏状态 `state` 会在每次迭代时传递，并根据输入数据进行更新。
4. **`l = loss(y_hat, y.long()).mean()`**：
    
    - 计算模型的损失 `l`。损失函数 `loss` 接收预测值 `y_hat` 和目标值 `y`，并将目标值 `y` 转换为 `long` 类型（通常用于分类任务）。
    - `mean()` 函数将损失的均值计算出来，因为可能在多个样本上计算损失。
5. **优化器类型判断 (`if isinstance(updater, torch.optim.Optimizer)`)**：
    
    - 这段代码根据优化器的类型选择不同的优化步骤：
        - **如果 `updater` 是 `torch.optim.Optimizer` 类型**（即 PyTorch 自带的优化器）：
            - 调用 `updater.zero_grad()` 清空之前的梯度。
            - 调用 `l.backward()` 进行反向传播，计算当前批次的梯度。
            - 调用 `grad_clipping(net, 1)` 对梯度进行裁剪，防止梯度爆炸。`1` 是裁剪的阈值。
            - 最后调用 `updater.step()` 进行优化步骤，更新模型参数。
        - **如果 `updater` 不是 `torch.optim.Optimizer` 类型**：
            - 仍然执行反向传播 `l.backward()` 和梯度裁剪 `grad_clipping(net, 1)`。
            - 这里没有使用 `updater.step()`，而是调用 `updater(batch_size=1)`，这种情况下 `updater` 可能是自定义的更新函数，它根据给定的批量大小进行模型参数的更新。
6. **`metric.add(l * y.numel(), y.numel())`**：
    
    - 这行代码是用来更新训练过程中的度量指标（如损失、准确率等）。
    - `l * y.numel()` 表示将当前批次的损失乘以目标 `y` 中的元素数量。`y.numel()` 返回目标 `y` 中元素的总数。
    - `metric.add()` 通常是一个度量工具，用于累加训练过程中的损失和样本数。

### 总结：

这段代码展示了典型的神经网络训练步骤，尤其是处理序列模型（如 RNN、LSTM、GRU）时的训练流程。整个过程包括：

1. **数据处理**：将输入和目标标签转换为合适的形状，并移动到指定设备。
2. **前向传播**：通过网络计算预测结果，并更新隐藏状态。
3. **损失计算**：使用损失函数计算损失值，并求均值。
4. **梯度计算和优化**：根据优化器类型执行梯度清零、反向传播、梯度裁剪以及参数更新。
5. **度量指标更新**：在训练过程中跟踪并更新相关的度量指标（如损失）。

此外，梯度裁剪的操作 (`grad_clipping`) 用于防止梯度爆炸，这在递归神经网络训练中是常见的技术，尤其是在处理较长的序列时。

## isinstance()
在 Python 中，`isinstance()` 是一个内置函数，用于检查某个对象是否是特定类或类型的实例。它的常用格式为：
```
isinstance(object, class_or_tuple)
```
- `object`: 要检查的对象。
- `class_or_tuple`: 要检查的类型或类型的元组。

如果 `object` 是给定类型（或元组中任何一个类型）的实例，`isinstance()` 返回 `True`，否则返回 `False`。
### 用法说明：

1. **检查单一类型**：
    
    - 如果只需要检查某个对象是否属于一个特定的类或类型，可以直接传入类名。例如：
```
    x = 5 
    if isinstance(x, int): 
	    print("x 是一个整数")
```
这里 `isinstance(x, int)` 返回 `True`，因为 `x` 是一个 `int` 类型的对象。
**检查多个类型**：

- `isinstance` 可以接收一个类型元组来检查对象是否属于多个类型中的一个。例如：
```
x = 5 
if isinstance(x, (int, float)): 
	print("x 是一个整数或浮点数")
```
这里 `isinstance(x, (int, float))` 检查 `x` 是否是 `int` 或 `float` 类型中的任意一种。