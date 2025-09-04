# model
import torch.optim

from modules_with_edge_features import *

class MainModel(nn.Module):    
    """
    atten_time:多头注意力机制的使用次数
    nfeats:输入特征的维度
    """
    def __init__(self,dr,lr,nlayers,lamda,alpha,atten_time,nfeats):
        super(MainModel, self).__init__()

        self.drop1 = nn.Dropout(p=dr)
        self.fc1 = nn.Linear(640*atten_time, 256)  # for attention，将输入维度 640 * atten_time 映射到 256 维

        self.drop2 = nn.Dropout(p=dr)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)  # 将 256 维特征映射到 1 维

        # 实例化 RGN_EGNN 模型，隐藏层维度为 512
        self.rgn_egnn = RGN_EGNN(nlayers=2, nfeat=nfeats, nhidden=512, nclass=1, dropout=dr,
                                 lamda=lamda, alpha=alpha, variant=True, heads=1)
        # 实例化 GCN 模型，隐藏层维度为 128
        self.rgn_gcn2 = RGN_GCN(nlayers=nlayers, nfeat=nfeats, nhidden=128, nclass=1,
                                dropout=dr,
                                lamda=lamda, alpha=alpha, variant=True, heads=1)

        # 多头注意力机制：存储多个 Attention_1 模块，每个模块的隐藏层维度为 512+128（RGN_EGNN+GCN），注意力头数为 16 个，数量为stten_time
        self.multihead_attention = nn.ModuleList([Attention_1(hidden_size=512+128, num_attention_heads=16) for _ in range(atten_time)]) 

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1e-16)


    # G：蛋白质结构图
    def forward(self, G, h, x,adj,efeats):

        h = torch.squeeze(h)  #去除 h 和 x 中维度为 1 的维度
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea1 = self.rgn_egnn(G, h, x,efeats)  # 使用 RGN_EGNN 模型提取特征 fea1
        print("fea1 shape:", fea1.shape)
        fea1 = torch.unsqueeze(fea1, dim=0)   # 在 fea1 的第 0 维添加一个维度

        fea2 = self.rgn_gcn2(h, adj)
        print("fea2 shape:", fea2.shape)
        fea2 = torch.unsqueeze(fea2, dim=0)

        fea = torch.cat([fea1,fea2],dim=2)   # 将 fea1 和 fea2 沿着第 2 维拼接
        print("fea shape:", fea.shape)

        # gated self-attention
        # 8 个 head，维度为 640
        attention_outputs = []
        for i in range(len(self.multihead_attention)):
            multihead_output, _ = self.multihead_attention[i](fea)
            print(f"multihead_output {i} shape:", multihead_output.shape)
            attention_outputs.append(multihead_output)
        embeddings = torch.cat(attention_outputs, dim=2)
        
        print("embeddings shape:", embeddings.shape)  
        # 输出：embeddings shape: torch.Size([1, 207, 5120])后，报错：
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1937x6205 and 2365x512)
        print()

        # # self-attention
        # attention_outputs = []
        # for i in range(len(self.multihead_attention)):
        #     multihead_output, _ = self.multihead_attention[i](fea,fea,fea)
        #     attention_outputs.append(multihead_output)
        # embeddings = torch.cat(attention_outputs, dim=2)

        out = self.drop1(embeddings)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out

