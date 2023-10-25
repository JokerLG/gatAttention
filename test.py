# 作   者: 花容
# 创建时间: 2023/9/19 22:16


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adjacency_matrix):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, adjacency_matrix)
        self.gc2 = GraphConvolution(hidden_dim, output_dim, adjacency_matrix)

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = self.gc2(x)
        return x

# 定义图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, adjacency_matrix):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.adjacency_matrix = adjacency_matrix

    def forward(self, x):
        support = torch.mm(x, self.weight)
        output = torch.mm(self.adjacency_matrix, support)
        return output

# 定义训练函数
def train(model, features, labels, adjacency_matrix, optimizer, criterion, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features)
        print(output.shape)
        loss = criterion(output, labels)
        print(labels.shape)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 示例数据
adjacency_matrix = torch.Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # 一个简单的邻接矩阵
features = torch.Tensor([[0.2, 0.5], [0.3, 0.7], [0.4, 0.9]])  # 示例节点特征
labels = torch.LongTensor([0, 1, 0])  # 示例标签


# 初始化模型、优化器和损失函数
model = GCN(input_dim=2, hidden_dim=16, output_dim=2, adjacency_matrix=adjacency_matrix)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练模型
train(model, features, labels, adjacency_matrix, optimizer, criterion, epochs=1)

