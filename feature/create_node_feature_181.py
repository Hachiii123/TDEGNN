import pickle
import numpy as np
import torch
import os

train_list=[]
seqanno= {}
Query_ids=[]
query_seqs=[]
query_annos=[]


def one_hot_encode(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = np.zeros((len(sequence), len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded_sequence[i, aa_to_int[aa]] = 1
    return encoded_sequence


def create_features(query_ids,test_path,pkl_path,topo_path,esm2_5120_path,ProtTrans_path):
    # 1.加载 one-hot 向量
    with open(test_path, 'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            if query_id[-1].islower():
                # query_id += query_id[-1]
                query_id = query_id[:-1] + query_id[-1].upper()
                print(query_id,'-'*1000)
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    # 2.加载 one-hot 向量（20 dim）,对 train_573 和 test_129 生成 one-hot 编码
    query_seqs_181 = []
    with open(test_path, 'r') as f1:
        text_181 = f1.readlines()
        for i in range(1, len(text_181), 3):     
            query_seq_181 = text_181[i].strip()
            # if query_seq_181 != 'RRNRRLSSASVYRYYLKRISMNIGTTGHVNGLSIAGNPEIMRAIARLSEQETYNWVTDYAPSHLAKEVVKQISGKYNIPGAYQGLLMAFAEKVLANYILDYKGEPLVEIHHNFLWELMQGFIYTFVRKDGKPVTVDMSKVLTEIEDALFKLVKK':
            query_seqs_181.append(query_seq_181)
    encoded_proteins = [one_hot_encode(sequence) for sequence in query_seqs_181]
                        
    # 3.加载残基特征
    PDNA_residue_load=open(pkl_path,'rb')
    PDNA_residue=pickle.load(PDNA_residue_load)
    
    # 4.加载残基拓扑特征(140 dim)
    topo_features = []
    paths_topo = []
    for pid in query_ids:
        file_path = os.path.join(topo_path, f"{pid}_topo.pt")
        paths_topo.append(file_path)
    for file_path in paths_topo:
        if os.path.exists(file_path):
            topo_feature = torch.load(file_path, map_location='cpu').numpy()
            topo_features.append(topo_feature)
        else:
            print(f"Warning: Topo feature not found for {file_path}")
            # 创建零填充矩阵（需知道序列长度）
            seq_len = len(seqanno[os.path.basename(file_path).split('_')[0]]['seq'])
            topo_features.append(np.zeros((seq_len, 140)))  # 140是拓扑特征维度



    # 5.加载 esm2-t33 特征（1280 dim）
    # ESM2_33 = []
    # paths_esm = []
    # for i in query_ids:
    #    file_paths = esm2_33_path + '{}'.format(i) + '.npy'
    #    paths_esm.append(file_paths)
    # for file_path in paths_esm:
    #    ESM2_33_embedding = np.load(file_path)
    #    ESM2_33.append(ESM2_33_embedding)
    
    # 5.加载 esm2-5120 特征（5120 dim）   
    ESM2_5120 = []
    paths_5120 = []
    for i in query_ids:  
        file_paths = esm2_5120_path + '{}'.format(i) + '.npy'
        paths_5120.append(file_paths)
    for file_path in paths_5120:
        # print(file_path)
        ESM2_5120_embedding = np.load(file_path,allow_pickle=True)
        ESM2_5120.append(ESM2_5120_embedding)


    # 6.加载 ProtTrans 特征
    ProTrans_1024=[]
    paths_1024 = []
    for i in query_ids:
        if i == '6wq2_aa':
            i = '6wq2_a'
        file_paths = ProtTrans_path + '{}'.format(i) + '.npy'
        paths_1024.append(file_paths)
    for file_path in paths_1024:
        ProTrans_1024_embedding = np.load(file_path, allow_pickle=True)
        ProTrans_1024.append(ProTrans_1024_embedding)


    data = {}
    for i in query_ids:
        data[i] = []
        if i == '6wq2_aa':
            i = '6wq2_a'
        residues = PDNA_residue[i]
        labels = seqanno[i]['anno']
        data[i].append({'features': residues,'label': labels})


    feature1=[]
    feature2=[]
    feature3=[]
    feature4 = []
    feature5 = []
    feature6 = []

    protein_labels=[]

    for i in query_ids:
        residues=data[i]
        feature1.append(residues[0]['features'])
        protein_labels.append((residues[0]['label']))

    for j in range(len(query_ids)):
        if 0 <= j < len(encoded_proteins):  # 确保 j 在有效范围内
            feature2.append(encoded_proteins[j])
        else:
            print(len(query_ids))
            print(len(encoded_proteins))
            print("警告：索引 j 超出了 encoded_proteins 的范围。")

        feature3.append(ESM2_5120[j])    # 5120 dim bert_5120
        # feature3.append(ESM2_33[j])
        feature4.append(ProTrans_1024[j])  # 1024 dim protrans
        # feature6.append(MSA_256[j])  # 256 dim MSA

    
    node_features = {}
    for i in range(len(query_ids)):
        pid = query_ids[i]
        node_features[pid] = {
            'seq': i+1,
            'residue_fea': feature1[i],
            'esm2_5120': feature3[i],
            'prottrans_1024': feature4[i],
            'one-hot': feature2[i],
            'topo_fea': topo_features[i],  # 新增拓扑特征
            'label': protein_labels[i]
        }
    return node_features


def create_dataset(query_ids,test_path,pkl_path,topo_path,esm2_33_path,ProtTrans_path,
                   residue,one_hot,esm2_33,prottrans,topo):
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

    # 打印调试信息
    print("Parameters of create_dataset function:")
    print("length of query_ids:",len(query_ids))
    print("test_path:",test_path)
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

    # all 702 protein information
    # query_ids,test_path,pkl_path,msa_256_path,esm2_5120_path,ProtTrans_path
    node_features = create_features(query_ids,test_path,pkl_path,topo_path,esm2_33_path,ProtTrans_path)

    for i in query_ids:
        protein = node_features[i]

        mat1 = (protein['residue_fea'])
        mat2 = (protein['one-hot'])
        mat3 = protein['topo_fea']
        mat4 = (protein['esm2_5120'])
        # mat4 = (protein['esm2_33'])
        mat5 = (protein['prottrans_1024'])
        # mat6 = (protein['msa_256'])

        mat3 = torch.Tensor(mat3)
        mat4 = torch.Tensor(mat4)
        mat4 = torch.squeeze(mat4)
        mat5 = torch.Tensor(mat5)
        mat5 = torch.squeeze(mat5)


        print("当前蛋白质ID:", i)
        print("mat1 shape:", mat1.shape)
        print("mat2 shape:", mat2.shape)
        print("mat4 shape:", mat4.shape)  
        print("mat5 shape:", mat5.shape)

        # 水平拼接所有特征
        features[i] = np.hstack((mat1, mat2, mat3, mat4, mat5))

        labels = protein['label']
        y.append(labels)

    for key in query_ids:
        if key == '6wq2_aa':
            key = '6wq2_a'
        X.append(features[key])


    return X,y




