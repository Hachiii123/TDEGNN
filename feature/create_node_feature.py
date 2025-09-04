import pickle
import numpy as np
import torch
import os

train_list=[]
seqanno= {}
Query_ids=[]
query_seqs=[]
query_annos=[]

# 生成蛋白质序列的 one-hot 编码：L x 20
def one_hot_encode(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = np.zeros((len(sequence), len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded_sequence[i, aa_to_int[aa]] = 1
    return encoded_sequence

# 整合多种蛋白质节点特征，包括：
# 1.序列编码：one-hot 编码 ,20 dim
# 2.预训练嵌入：ESM2-t33(1280 dim)/ESM2-t48(5210 dim), ProtTrans 1024 dim
# 3.残基特征：PSSM,AF,DSSP 71 dim
# 4.残基拓扑特征：140 dim
# 5.标签：蛋白质的标注信息（FASTA文件第三行的01字符串，表示是否是结合位点）
def create_features(query_ids, all_702_path, train_path, test_path, pkl_path, topo_path,   
                    esm2_33_path, ProtTrans_path):  # 新增 topo_path

    # 1.加载蛋白质 ID,序列，标签
    with open(train_path,'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}  
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    with open(test_path, 'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    # 2.加载 one-hot 向量（20 dim）,对 train_573 和 test_129 生成 one-hot 编码
    #query_seqs_702 = []
    #with open(all_702_path, 'r') as f1:
    #    text_702 = f1.readlines()
    #    for i in range(1, len(text_702), 3):      
    #        query_seq_702 = text_702[i].strip()
    #        query_seqs_702.append(query_seq_702)
    #encoded_proteins = [one_hot_encode(sequence) for sequence in query_seqs_702]


    
    # 3.加载残基特征(71 dim)：PSSM,AF,DSSP
    PDNA_residue_load=open(pkl_path,'rb')
    PDNA_residue=pickle.load(PDNA_residue_load)


    # 4.加载残基拓扑特征(140 dim)            
    # topo_features = []
    # paths_topo = []
    # for pid in query_ids:
    #     file_path = os.path.join(topo_path, f"{pid}_topo.pt")
    #     paths_topo.append(file_path)
    # for file_path in paths_topo:
    #     if os.path.exists(file_path):
    #         topo_feature = torch.load(file_path, map_location='cpu').numpy()
    #         topo_features.append(topo_feature)
    #     else:
    #         print(f"Warning: Topo feature not found for {file_path}")
    #         # 创建零填充矩阵（需知道序列长度）
    #         seq_len = len(seqanno[os.path.basename(file_path).split('_')[0]]['seq'])
    #         topo_features.append(np.zeros((seq_len, 140)))  # 140是拓扑特征维度

    # 4.加载经过噪声扰动后的残基拓扑特征(140 dim) 
    topo_features = []
    paths_topo = []
    for pid in query_ids[573:]:
        file_path = os.path.join(topo_path, f"{pid}_noise0.5_topo.pt")
        paths_topo.append(file_path)
    for file_path in paths_topo:
        if os.path.exists(file_path):
            loaded_data = torch.load(file_path, map_location='cpu')

            if isinstance(loaded_data, dict):  # 如果是包含特征的字典
                feature_array = loaded_data['features'].numpy()
            else:  # 如果是直接的特征张量
                feature_array = loaded_data.numpy() if torch.is_tensor(loaded_data) else loaded_data
            
            # 维度检查
            if feature_array.shape[1] != 140:
                raise ValueError(f"特征维度应为140，但得到{feature_array.shape[1]}")
            
            topo_features.append(feature_array)
        
        else:
            print(f"Warning: Topo feature not found for {file_path}")
            # 创建零填充矩阵（需知道序列长度）
            pid = os.path.basename(file_path).split('_noise')[0]
            seq_len = len(seqanno.get(pid, {}).get('seq', '')) or 100
            topo_features.append(np.zeros((seq_len, 140)))  # 140是拓扑特征维度
             

    # 5.加载 esm2-t33 特征（1280 dim）
    ESM2_33 = []
    paths = []
    for i in query_ids:
        file_paths = esm2_33_path + '{}'.format(i) + '.npy'
        paths.append(file_paths)
    for file_path in paths:
        ESM2_33_embedding = np.load(file_path, allow_pickle=True)
        ESM2_33.append(ESM2_33_embedding)

    # 5.加载 esm2-t48 特征（5120 dim）
    # ESM2_5120 = []
    # paths_5120 = []
    # for i in query_ids:
        # file_paths = esm2_5120_path + '{}'.format(i) + '.rep_5120.npy'  
    #   file_paths = esm2_5120_path + '{}'.format(i) + '.npy'
    #   paths_5120.append(file_paths)
    # for file_path in paths_5120:
        # print(file_path)
    #   ESM2_5120_embedding = np.load(file_path,allow_pickle=True)
    #   ESM2_5120.append(ESM2_5120_embedding)

    # 6.加载 ProtTrans 特征（1024 dim）
    ProTrans_1024=[]
    paths_1024 = []
    for i in query_ids:
        file_paths = ProtTrans_path + '{}'.format(i) + '.npy'
        paths_1024.append(file_paths)
    for file_path in paths_1024:
        ProTrans_1024_embedding = np.load(file_path, allow_pickle=True)
        ProTrans_1024.append(ProTrans_1024_embedding)

    # load residue features-71dim and labels
    data = {}
    for i in query_ids:
        # print("当前蛋白质：", i)
        data[i] = []
        residues = PDNA_residue[i]
        labels = seqanno[i]['anno']
        data[i].append({'features': residues, 'label': labels})


    feature1=[]
    feature2=[]
    feature3=[]
    feature4 = []
    protein_labels=[]

    #for i in query_ids:
    #    residues=data[i]
    #    feature1.append(residues[0]['features'])
    #    protein_labels.append((residues[0]['label']))
    #    seq = seqanno[i]['seq']
    #    onehot = one_hot_encode(seq)
    #    feature2.append(onehot)

    #for j in range(len(query_ids)):
    #    #feature2.append(encoded_proteins[j])   # 20 dim one-hot
    #    feature3.append(ESM2_33[j])            # 1280 dim esm2_t33
    #    # feature3.append(ESM2_5120[j])        # 5120 dim esm2_5120
    #    feature4.append(ProTrans_1024[j])      # 1024 dim protrans
    
    for i,pid in enumerate(query_ids):
        residues = data[pid]
        feature1.append(residues[0]['features'])
        protein_labels.append(residues[0]['label'])
        
        seq = seqanno[pid]['seq']
        onehot = one_hot_encode(seq)
        feature2.append(onehot)
        
        feature3.append(ESM2_33[i])                                # esm2_t33
        feature4.append(ProTrans_1024[i])  
                
        
          
    # 加入拓扑特征后的节点特征构建
    # 构建node_features时加入拓扑特征
    node_features = {}
    for i, pid in enumerate(query_ids):
        node_features[pid] = {
            'seq': i + 1,
            'residue_fea': feature1[i],
            'esm2_33': feature3[i],
            'prottrans_1024': feature4[i],
            'one-hot': feature2[i],
            #'topo_fea': topo_features[i],
            'label': protein_labels[i]
        }

    return node_features



def create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,topo_path,esm2_33_path,
                   ProtTrans_path,residue,one_hot,esm2_33,prottrans,topo):   # 新增 topo_path，topo 参数
    '''
    :param query_ids: all protein ids
    :param train_path: training set file path
    :param test_path: test_129 set file path
    :param all_702_path: train_573 and test_129 file path
    :param pkl_path: residue features path
    :param esm2_33_path: esm2-t36 embeddings path
    :param esm2_5120_path: esm2-t48 embeddings path
    :param ProtTrans_path: ProtTrans embeddings path
    :param residue: add residue features or not
    :param one_hot: add one-hot features or not
    :param esm2_33: add esm2-t36 features or not
    :param esm_5120: add esm2-t48 features or not
    :param prottrans: add ProtTrans features or not
    :return: X and y, involving training and validation set
    '''

    print("Parameters of create_dataset function:")
    print("length of query_ids:",len(query_ids))
    print("train_path:",train_path)
    print("test_path:",test_path)
    print("all_702_path:",all_702_path)
    print("pkl_path:",pkl_path)
    print("topo_path:",topo_path)
    print("esm2_33_path:",esm2_33_path)
    print("ProtTans_path:",ProtTrans_path)
    print("residue:",residue)
    print("one_hot:",one_hot)
    print("esm2_33:",esm2_33)
    print("prottrans:",prottrans)
    print("topo:",topo)


    X=[]
    y=[]
    features={}

    node_features = create_features(query_ids,all_702_path,train_path,test_path,pkl_path,topo_path,esm2_33_path,ProtTrans_path)

    uncorrect_prot=[]  # 特征维度不正确的蛋白质
    
    for i in query_ids:
        protein = node_features[i]
        mat1 = (protein['residue_fea'])
        mat2 = (protein['one-hot'])
        #mat3 = protein['topo_fea']   # 新增拓扑特征
        mat4 = (protein['esm2_33'])
        mat5 = (protein['prottrans_1024'])

        #mat3 = torch.Tensor(mat3)
        mat4 = torch.Tensor(mat4)
        mat4 = torch.squeeze(mat4)
        mat5 = torch.Tensor(mat5)
        mat5 = torch.squeeze(mat5)



        # 报错：np.hstack()时行数不匹配，检查报错的蛋白质的特征维度
        # 对特征维度不正确的蛋白质，重新提取esm2_t33的特征？        
        if mat4.shape[1] != 1280:
            uncorrect_prot.append(i)
        
        
        if residue == True and one_hot == True and topo == True and esm2_33 == True and prottrans == True:      # all embeddings
            features[i] = np.hstack((mat1, mat2, mat3, mat4, mat5))
            #print("Training TDEGNN_full")
        elif residue == True and one_hot == True and topo == False and esm2_33 == True and prottrans == True:   # w/o topo features
            features[i] = np.hstack((mat1, mat2, mat4, mat5))
            print("Training TDEGNN_no_topo")
        elif residue == False and one_hot == False and topo == True and esm2_33 == False and prottrans == False:  # only topo features 
            features[i] = np.hstack((mat3))
            print("Training TDEGNN_only_topo")
        elif residue == False and one_hot == False and topo == False and esm2_33 == True and prottrans == True:  # only protein language model embeddings
            features[i] = np.hstack((mat4, mat5))
            print("Training TDEGNN_only_pLM")

        

        # 水平拼接所有特征
        # features[i] = np.hstack((mat1, mat2, mat3, mat4, mat5))

        labels = protein['label']
        y.append(labels)

    print("nums of uncorrect_prot:",len(uncorrect_prot))
    print()

    for key in query_ids:
        X.append(features[key])
        
                    

    return X,y







