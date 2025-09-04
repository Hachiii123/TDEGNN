# create edge_features of protein graphs on DNA_573_Train and DNA_129_Test

import pickle
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser
import os
import logging


# 设定阈值为17
# 论文中 Structural context extraction:目标残基的结构性上下文由以残基为中心，半径为17埃的滑动球体决定
# th 用于判断两个氨基酸残基之间是否存在边。如果两个残基之间的距离 <th,则认为它们之间存在边
th = 17   

# 构建氨基酸残基距离矩阵列表
# dis_path：PDNA_psepos_SC.pkl, 伪位置
# Query_ids：数据集中每个蛋白质的 PDB ID
def create_dis_matrix(dis_path,Query_ids):
    print("当前创建距离矩阵的文件路径：",dis_path)
    print("当前蛋白质数量：",len(Query_ids))
    
    dis_load=open(dis_path,'rb')        # 加载残基位置数据
    dis_residue=pickle.load(dis_load)   # 字典类型，键是蛋白质 ID，值是残基坐标列表

    distance_matrixs=[]  # 所有蛋白质的距离矩阵

    for i in Query_ids:
        if i not in dis_residue:
            logging.warning(f"PDB ID {i} not found in dis_residue.")
            continue
        
        residues=dis_residue[i]   
        num_node = len(residues)
        residues_array = np.array(residues)
        distance_matrix = np.zeros((num_node, num_node))    # 初始化当前蛋白质的 [残基数 x 残基数] 的残基距离矩阵
        distances = np.linalg.norm(residues_array[:, np.newaxis, :] - residues_array[np.newaxis, :, :], axis=-1)   # 计算残基之间的欧几里得距离
        distance_matrix[np.triu_indices(num_node, k=1)] = distances[np.triu_indices(num_node, k=1)]
        distance_matrix += distance_matrix.T       # 将距离矩阵的上三角部分复制到下三角
        distance_matrixs.append(distance_matrix)   # 添加到所有蛋白质距离矩阵的列表中

    # 检查query_ids长度与dis_matrix长度是否一致 
    print(f"distance_matrixs 的长度: {len(distance_matrixs)}")
    
    return distance_matrixs



def parse_residue_coordinates(pdb_file_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('protein', pdb_file_path)
    model = structure[0]  # 选择第一个模型
    residues = []

    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != ' ':  # 跳过非标准残基
                continue
            # 获取 CA 原子的坐标
            if 'CA' in residue:
                residues.append(residue['CA'].get_coord())

    return np.array(residues)


def create_dis_matrix_by_pdb(pdb_file_path):
    # 从 PDB 文件中解析残基坐标
    residues_array = parse_residue_coordinates(pdb_file_path)

    num_node = len(residues_array)
    distance_matrix = np.zeros((num_node, num_node))

    # 计算残基之间的欧氏距离
    distances = np.linalg.norm(residues_array[:, np.newaxis, :] - residues_array[np.newaxis, :, :], axis=-1)
    distance_matrix[np.triu_indices(num_node, k=1)] = distances[np.triu_indices(num_node, k=1)]
    distance_matrix += distance_matrix.T 

    return distance_matrix



# 根据 th 生成邻接矩阵，得到边的索引列表
# dis_matrix：距离矩阵，protein_idx：蛋白质 id，th: 阈值
def cal_edges(dis_matrix,protein_idx,th):
    
    dis_matrix_copy = dis_matrix.copy()

    binary_matrix = (dis_matrix_copy[protein_idx] <= th).astype(int)         # 根据阈值将距离矩阵转换为二进制矩阵（<=th为1，>th为0）
    symmetric_matrix = np.triu(binary_matrix) + np.triu(binary_matrix, 1).T  # 转换为对称矩阵
    dis_matrix_copy[protein_idx] = symmetric_matrix
    binary_matrix = torch.from_numpy(dis_matrix_copy[protein_idx])           # 将二进制矩阵转换为 PyTorch 张量

    mask = (binary_matrix ==1)
    radius_index_list = np.where(mask)  
    radius_index_list = [list(nodes) for nodes in zip(radius_index_list[0], radius_index_list[1])]  # 找到二进制矩阵中值为 1 的位置，生成边的索引列表 radius_index_list

    return radius_index_list


# 计算边的属性：距离，余弦相似度
# edge_index_list：边的索引列表，distance_matrix：距离矩阵
def calculate_edge_attributes(edge_index_list, distance_matrixs, protein_idx):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)
    cossim = nn.CosineSimilarity(dim=0)

    num_edges = len(edge_index_list) 
    distance_attrs = torch.zeros(num_edges)  # 初始化距离属性和余弦相似度属性
    cos_similarity_attrs = torch.zeros(num_edges)

    # 遍历每条边，计算两个残基之间的距离和余弦相似度
    for i in range(num_edges):
        src_idx, dst_idx = edge_index_list[i]

        distance_matrix_src = torch.tensor(distance_matrixs[protein_idx][src_idx][0])
        distance_matrix_dst = torch.tensor(distance_matrixs[protein_idx][dst_idx][0])
        distance = pdist(distance_matrix_src, distance_matrix_dst).item()
        distance_attrs[i] = distance / 17    # 将距离归一化到 [0, 1] 范围

        distance_matrix_src_array = torch.tensor(distance_matrixs[protein_idx][src_idx])
        distance_matrix_dst_array = torch.tensor(distance_matrixs[protein_idx][dst_idx])
        cos_similarity = cossim(distance_matrix_src_array, distance_matrix_dst_array).item()
        cos_similarity_attrs[i] = (cos_similarity + 1) / 2  # 将余弦相似度归一化到 [0, 1] 范围

    return distance_attrs, cos_similarity_attrs


# 获取训练集（573）的边的属性
def get_edge_attr_train(pro_id,th,distance_matrixs):
    edge_index_list = cal_edges(distance_matrixs[:573], protein_idx=pro_id, th=th)
    distance_attrs, cos_similarity_attrs = calculate_edge_attributes(edge_index_list, distance_matrixs[:573], protein_idx=pro_id)
    edge_attr_train = torch.stack((distance_attrs, cos_similarity_attrs), dim=1)

    return edge_attr_train


# 获取测试集（129）的边的属性
def get_edge_attr_test(pro_id,th,distance_matrixs):
    edge_index_list = cal_edges(distance_matrixs, protein_idx=pro_id, th=th)
    distance_attrs, cos_similarity_attrs = calculate_edge_attributes(edge_index_list, distance_matrixs, protein_idx=pro_id)
    edge_attr_test = torch.stack((distance_attrs, cos_similarity_attrs), dim=1)

    return edge_attr_test


# 计算edge的时候放出来，train的时候注释掉
root_dir = '/home/duying/EGPDI/data/'
train_path = root_dir + 'DNA-573_Train.txt'
test_path = root_dir + 'DNA-129_Test.txt'

# 无需加载 PDB，计算特征和训练时都注释掉
# pdb_folder_path = '/home/duying/EGPDI/data/PDB/'
# pdb_folder_path = '/home/duying/EGPDI/data/AF3PDB/'


# 伪位置文件路径 ，换数据集记得换这里的路径
# dis_path= root_dir + 'dataset_dir_181/PDNA_psepos_SC.pkl'
dis_path= root_dir + 'AF2_residue_feat_702/PDNA_psepos_SC.pkl'
query_ids = []


# 读取 PDB ID
# test_path/train_path
with open(train_path, 'r') as f:
    text = f.readlines()
    for i in range(0, len(text), 3):
        query_id = text[i].strip()[1:]
        query_ids.append(query_id)


# 计算距离矩阵
# distance_matrixs=create_dis_matrix_by_pdb(pdb_folder_path)
distance_matrixs=create_dis_matrix(dis_path,query_ids)


# 遍历训练集中每个蛋白质，计算边的特征
# get_edge_attr_test/get_edge_attr_train
efeats = []
for pro_id in range(len(query_ids)):
    egde_feats = get_edge_attr_train(pro_id,th,distance_matrixs)
    efeats.append(egde_feats)
    print("正在计算")

#save_edgefeats_path = '/home/duying/EGPDI/data/AF2_Edge_feat/test_129/EdgeFeats_predicted_SC_17_129.pkl'
#save_edgefeats_path = '/home/duying/EGPDI/data/AF3_Edge_feat/test_181/EdgeFeats_predicted_SC_17_181.pkl'
save_edgefeats_path = '/home/duying/EGPDI/data/AF2_Edge_feat/train_573/EdgeFeats_predicted_SC_17_573.pkl'


if not os.path.exists(save_edgefeats_path):
    os.makedirs("/home/duying/EGPDI/data/AF2_Edge_feat/train_573")
    with open(save_edgefeats_path, 'wb') as f:
        pickle.dump(efeats, f)  
        print("已保存至：" + save_edgefeats_path)