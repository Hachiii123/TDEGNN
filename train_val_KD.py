
import os
import time
import pickle
import dgl
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from sklearn.model_selection import KFold

from model_KD import MainModel
from feature.create_node_feature import create_dataset
from feature.create_graphs import get_coor_train,get_adj_predicted
from feature.create_edge import create_dis_matrix,get_edge_attr_train

import warnings
warnings.filterwarnings("ignore")
seed_value = 1995
th=17

cuda_index = 1  # 这里可以设置为 0, 1, 2, ... 对应不同的 GPU

# 检查是否有足够的 GPU 设备
if torch.cuda.is_available() and cuda_index < torch.cuda.device_count():
    device = torch.device(f"cuda:{cuda_index}")
else:
    device = torch.device("cpu")
    print("CUDA is not available or the specified index is out of range. Falling back to CPU.")


# 模型保存路径
Model_Path = '/home/duying/EGPDI/models_TDEGNN_noTopo'

features = []
labels = []

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


root_dir = '/home/duying/EGPDI/data/'
train_path= root_dir + 'DNA-573_Train.txt'
test_path= root_dir + 'DNA-129_Test.txt'
all_702_path = root_dir +  'DNA-702.txt'

# AF2 
# pkl_path= root_dir + 'PDNA_residue_feas_PSA.pkl' 
# topo_path = root_dir + 'topo_features/'
# esm2_33_path = root_dir + 'ESM2_t33/'       # .npy文件，883个
# esm2_5120_path= root_dir + 'ESM2_t48/'      
# ProtTrans_path = root_dir + 'ProtTrans/'    # prtoTrans embedding (.npy文件，883个)
# dis_path= root_dir + 'PDNA_psepos_SC.pkl'

# AF3
pkl_path= root_dir + 'AF3_residue_feats/train_test129/PDNA_residue_feas_PSA.pkl'
topo_path = root_dir + 'AF3_topo_features/'
esm2_33_path = root_dir + 'ESM2_t33/'
#esm2_5120_path= root_dir + 'ESM2_t48/'
ProtTrans_path = root_dir + 'ProtTrans/'
dis_path= root_dir + 'AF3_residue_feats/train_test129/PDNA_psepos_SC.pkl'


query_ids = []

#读取所有训练集的id
with open(train_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_ids.append(query_id)

# 训练时注释掉
# with open(test_path, 'r') as f:
#     train_text = f.readlines()
#     for i in range(0, len(train_text), 3):
#         query_id = train_text[i].strip()[1:]
#         query_ids.append(query_id)

print(f"query_ids长度:",len(query_ids))

# 创建数据集
X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,topo_path,esm2_33_path,
                     ProtTrans_path,residue=True,one_hot=True,esm2_33=True,prottrans=True,topo=False)
distance_matrixs=create_dis_matrix(dis_path,query_ids)


X_train = X[:573]
X_test = X[573:]

y_train = y[:573]
y_test = y[573:]

NUMBER_EPOCHS = 30

# final model parameters
dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

# 自蒸馏参数
distill_temp = 4.0     # 蒸馏温度参数
alpha_ce = 0.5         # 交叉熵损失权重
alpha_kd = 0.5         # 知识蒸馏损失权重
alpha_fd = 0.5         # 特征蒸馏损失权重

IDs = query_ids[:573]
sequences = []
labels = []

with open(all_702_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences[:573]

labels = y_train
features = X_train
coors = get_coor_train(dis_path, query_ids)
adjs = get_adj_predicted(IDs)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)


save_edgefeats_path = root_dir + 'AF3_Edge_feat/train_573/EdgeFeats_predicted_SC_17_573.pkl'
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

# features = X_train，X_train = X[:573]，X 来自create_dataset
print(len(IDs), len(sequences), len(labels), len(features), len(coors), len(adjs), len(graphs), len(efeats))


train_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs,'efeats':efeats}
dataframe = pd.DataFrame(train_dic)


class dataSet(data.Dataset):

    def __init__(self,dataframe,adjs):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.features = dataframe['features'].values
        self.coors = dataframe['coors'].values     #坐标
        self.graphs =  dataframe['graph'].values   #邻接构建的图
        self.efeats = dataframe['efeats'].values   #边特征
        self.adj = dataframe['adj'].values         #邻接


    def __getitem__(self,index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        node_features = self.features[index]
        coors = self.coors
        coor = coors[index]
        graphs = self.graphs
        graph = graphs[index]
        adj = self.adj[index]

        efeat = self.efeats[index]

        return sequence_name,sequence,label,node_features,graph,efeat,adj,coor

    def __len__(self):

        return len(self.labels)


def graph_collate(samples):
    ID_batch, sequence_batch,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch = map(list, zip(*samples))
    graph_batch = dgl.batch(graph_batch)

    return ID_batch, sequence_batch,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch


# 知识蒸馏损失函数
def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """
    计算知识蒸馏损失
    outputs: 学生模型输出
    teacher_outputs: 教师模型输出
    temperature: 温度参数，用于软化概率分布
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature**2)


# 特征蒸馏损失函数
def feature_distillation_loss(student_features, teacher_features):
    """
    计算特征蒸馏损失，使用MSE损失度量特征表示之间的差异
    student_features: 学生模型特征
    teacher_features: 教师模型特征
    """
    # 对特征进行归一化
    student_features_norm = F.normalize(student_features, p=2, dim=1)
    teacher_features_norm = F.normalize(teacher_features, p=2, dim=1)
    
    # 计算MSE损失
    return F.mse_loss(student_features_norm, teacher_features_norm)


def train_one_epoch(model,data_loader):
    epoch_loss_train = 0.0
    n = 0
    count_prot = 0

    # 设置设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移动到设备

    # 添加ID-batch和sequence_batch
    for ID_batch, sequence_batch,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:
        # 使用esm2-t33时，跳过 3cmu_A
        if ID_batch == "3cmu_A":
            print(f"跳过蛋白质 {ID_batch}")
            continue

        model.optimizer.zero_grad()

        # 将数据转换为张量
        node_features_batch = torch.tensor(node_features_batch)
        coors_batch = torch.tensor(coors_batch)
        adj_batch = adj_batch[0]
        y_true = label_batch[0]
        efeat_batch = efeat_batch[0]

        #if torch.cuda.is_available():
        #    node_features_batch = Variable(node_features_batch.cuda())
        #    graph_batch = graph_batch.to(device)
        #    efeat_batch = efeat_batch.to(device)
        #    adj_batch = Variable(adj_batch.cuda())
        #    coors_batch = Variable(coors_batch.cuda())
        #    y_true = label_batch
        #else:
        #    node_features_batch = Variable(node_features_batch)
        #    graph_batch = graph_batch
        #    adj_batch = Variable(adj_batch)
        #    coors_batch = Variable(coors_batch)
        #    y_true = label_batch
        #    efeat_batch = efeat_batch
        
        
        # 移动到设备
        device = next(model.parameters()).device
        node_features_batch = Variable(node_features_batch.to(device))
        graph_batch = graph_batch.to(device)
        efeat_batch = efeat_batch.to(device)
        adj_batch = Variable(adj_batch.to(device))
        coors_batch = Variable(coors_batch.to(device))

        count_prot+=1
        if count_prot % 50 == 0:  # 每50个样本输出一次信息，避免过多输出
            print(f"Sample {count_prot}:")
            print(f"ID: {ID_batch}")
            print('node_features', node_features_batch.shape)    
            print('graph', graph_batch)         
            print('adj', adj_batch.shape)       
            print('coors', coors_batch.shape)   
            print('y', len(y_true))
            print('efeats', efeat_batch.shape) 
        
        

        try:
            # 前向传播，获取多层预测结果和特征表示
            outputs, features = model(graph_batch, node_features_batch, coors_batch, adj_batch, efeat_batch)

            # 解包输出，outputs[0]是最深层输出，outputs[1]和outputs[2]是浅层输出
            final_output, middle_output, shallow_output = outputs
            final_features, middle_features, shallow_features = features
            

            # 转换为概率分布
            final_probs = torch.sigmoid(final_output)
            middle_probs = torch.sigmoid(middle_output)
            shallow_probs = torch.sigmoid(shallow_output)

            # 准备标签
            labels = torch.tensor([int(l) for l in y_true], dtype=torch.float32, device=device)
            assert labels.shape == final_output.shape, f"标签维度 {labels.shape} 与输出 {final_output.shape} 不一致"

            # 分别计算每层的监督损失
            final_ce_loss   = F.binary_cross_entropy_with_logits(final_output, labels)
            middle_ce_loss  = F.binary_cross_entropy_with_logits(middle_output, labels)
            shallow_ce_loss = F.binary_cross_entropy_with_logits(shallow_output, labels)

            # 计算知识蒸馏损失 (从深层到浅层)
            middle_kd_loss = kd_loss(middle_output, final_output.detach(), temperature=distill_temp)
            shallow_kd_loss = kd_loss(shallow_output, final_output.detach(), temperature=distill_temp)
            
            # 计算特征蒸馏损失
            middle_fd_loss = feature_distillation_loss(middle_features, final_features.detach())
            shallow_fd_loss = feature_distillation_loss(shallow_features, final_features.detach())

            # 计算总损失
            # 1. 最深层只有监督损失
            final_loss = final_ce_loss
            
            # 2. 中层结合监督损失、知识蒸馏损失和特征蒸馏损失
            middle_loss = alpha_ce * middle_ce_loss + alpha_kd * middle_kd_loss + alpha_fd * middle_fd_loss
            
            # 3. 浅层结合监督损失、知识蒸馏损失和特征蒸馏损失
            shallow_loss = alpha_ce * shallow_ce_loss + alpha_kd * shallow_kd_loss + alpha_fd * shallow_fd_loss
            
            # 总损失是所有层损失的加权和
            total_loss = final_loss + middle_loss + shallow_loss
        
        except Exception as e:
            print(f"[跳过异常蛋白质] ID: {ID_batch} 报错: {e}")
            continue
        
        # 反向传播和优化
        total_loss.backward()
        model.optimizer.step()

        epoch_loss_train += total_loss.item()
        n += 1

    print('training time',n)
    epoch_loss_train_avg = epoch_loss_train / n

    return epoch_loss_train_avg


def evaluate(model,data_loader):

    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = []

    for ID_batch, sequence_batch,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:
        
        #print("当前蛋白质ID：",ID_batch)
        if ID_batch == "3cmu_A":
            print(f"跳过蛋白质 {ID_batch}")
            continue
        try:    
            with torch.no_grad():
    
                node_features_batch = torch.tensor(node_features_batch)
                coors_batch = torch.tensor(coors_batch)
                adj_batch = adj_batch[0]
                label_batch = label_batch[0]
                efeat_batch = efeat_batch[0]
    
                if torch.cuda.is_available():
                    node_features_batch = Variable(node_features_batch.cuda())
                    graph_batch = graph_batch.to(device)
                    efeat_batch = efeat_batch.to(device)
                    adj_batch = Variable(adj_batch.cuda())
                    coors_batch = Variable(coors_batch.cuda())
                    y_true = label_batch
                else:
                    node_features_batch = Variable(node_features_batch)
                    graph_batch = graph_batch
                    adj_batch = Variable(adj_batch)
                    coors_batch = Variable(coors_batch)
                    y_true = label_batch
                    efeat_batch = efeat_batch
    
                # 只使用最终输出进行评估
                outputs, _ = model(graph_batch, node_features_batch, coors_batch, adj_batch, efeat_batch)
                y_pred = outputs[0]  # 使用最深层的输出作为最终预测
                
                # 应用sigmoid激活函数获取概率
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.squeeze(y_pred)
    
    
                y_true_int = [int(label) for label in y_true]
                y_true = torch.tensor(y_true_int, dtype=torch.float32, device=device)
    
                loss = model.criterion(y_pred,y_true)
    
                y_pred = y_pred.cpu().detach().numpy()
                y_true = y_true.cpu().detach().numpy()
    
                valid_pred += [pred for pred in y_pred]
                valid_true += list(y_true)
    
                epoch_loss += loss.item()
                n += 1
                
        except Exception as e:
            print(f"[跳过异常蛋白质] ID: {ID_batch}, 报错信息: {e}")
            continue     

    epoch_loss_avg = epoch_loss / n
    print('evaluate time', n)

    return epoch_loss_avg,valid_true,valid_pred


def analysis(y_true,y_pred,best_threshold = None):

    if best_threshold == None:
        best_mcc = 0
        best_threshold = 0

        for j in range(0, 100):
            # threshold = j / 100000  # pls change this threshold according to your code
            threshold = j / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

    print('best_threshold',best_threshold)
    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]
    
    
    correct_samples = sum(a == b for a, b in zip(binary_pred, y_true))
    accuracy = correct_samples / len(y_true)

    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)
    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    results = {
        'accuracy':accuracy,
        'spe':spe,
        'precision': pre,
        'recall': recall,
        'f1':f1,
        'mcc': mcc,
        'auc':auc,
        'pr_auc':pr_auc,
        'thred':best_threshold,
        'tn': tn,
        'tp': tp,
        'fn': fn,
        'fp': fp
    }

    return results


def train_1(model,train_dataframe,valid_dataframe,fold = 0):

    train_dataSet = dataSet(dataframe=train_dataframe, adjs=adjs)
    train_loader = torch.utils.data.DataLoader(train_dataSet,batch_size=1,shuffle=True,collate_fn=graph_collate)

    valid_dataSet = dataSet(dataframe=valid_dataframe, adjs=adjs)
    valid_loader = torch.utils.data.DataLoader(valid_dataSet, batch_size=1, shuffle=True, collate_fn=graph_collate)

    # 初始化最佳指标跟踪
    best_epoch = 0
    best_val_acc = 0
    best_val_spe = 0
    best_val_pre = 0
    best_val_recall = 0
    best_val_f1 = 0
    best_val_mcc = 0
    best_val_auc = 0
    best_val_prauc = 0
    best_val_tn = 0
    best_val_tp = 0
    best_val_fn = 0
    best_val_fp = 0

    for epoch in range(NUMBER_EPOCHS):

        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        begin_time = time.time()
        
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        end_time = time.time()
        run_time = end_time - begin_time

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred = evaluate(model, valid_loader)
        valid_results = analysis(valid_true, valid_pred)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid accuracy: ", valid_results['accuracy'])
        print("Valid spe: ", valid_results['spe'])
        print("Valid precision: ", valid_results['precision'])
        print("Valid recall: ", valid_results['recall'])
        print("Valid f1: ", valid_results['f1'])
        print("Valid mcc: ", valid_results['mcc'])
        print("Valid auc: ", valid_results['auc'])
        print("Valid pr_auc: ", valid_results['pr_auc'])
        print("Running Time: ", run_time)

        if best_val_prauc < valid_results['pr_auc']:
            best_epoch = epoch + 1
            best_val_mcc = valid_results['mcc']
            best_val_acc = valid_results['accuracy']
            best_val_spe = valid_results['spe']
            best_val_pre = valid_results['precision']
            best_val_recall = valid_results['recall']
            best_val_f1 = valid_results['f1']
            best_val_auc = valid_results['auc']
            best_val_prauc = valid_results['pr_auc']
            best_val_tn = valid_results['tn']
            best_val_tp = valid_results['tp']
            best_val_fn = valid_results['fn']
            best_val_fp = valid_results['fp']
            

            print('-' * 20, "new best pr_auc:{0}".format(best_val_prauc), '-' * 20)
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + 'predicted_edgeFeats_best_AUPR_model.pkl'))

    return best_epoch,best_val_mcc,best_val_acc,best_val_spe,best_val_pre,best_val_recall,best_val_f1,best_val_auc,best_val_prauc,best_val_tn,best_val_tp,best_val_fn,best_val_fp


def cross_validation(all_dataframe,fold_number = 5):

    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    best_epochs = []
    valid_accs = []
    valid_spes = []
    valid_recalls = []
    valid_mccs = []
    valid_f1s = []
    valid_pres = []
    valid_aucs = []
    valid_pr_aucs = []
    valid_tns = []
    valid_tps = []
    valid_fns = []
    valid_fps = []    

    for train_index,valid_index in kfold.split(sequence_names,sequence_labels):

        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]),"samples, validate on",str(valid_dataframe.shape[0]),"samples")


        # PSSM 20,Atomic features 7, SS 14 
        # one-hot 20 
        # protrans 1024 
        # topo 140
        # ESM2 1280/5120
        # model = MainModel(dr,lr,nlayers,lamda,alpha,atten_time,nfeats=41+20+140+5120/1280+1024)
        model = MainModel(dr,lr,nlayers,lamda,alpha,nfeats=41+20+1280+1024)


        best_epoch,valid_mcc,val_acc,val_spe,val_pre,val_recall,val_f1,val_auc,val_pr_auc,val_tn,val_tp,val_fn,val_fp = train_1(model,train_dataframe,valid_dataframe,fold+1)
        best_epochs.append(str(best_epoch))
        valid_mccs.append(valid_mcc)
        valid_accs.append(val_acc)
        valid_spes.append(val_spe)
        valid_pres.append(val_pre)
        valid_recalls.append(val_recall)
        valid_f1s.append(val_f1)
        valid_aucs.append(val_auc)
        valid_pr_aucs.append(val_pr_auc)
        valid_tns.append(val_tn)
        valid_tps.append(val_tp)
        valid_fns.append(val_fn)
        valid_fps.append(val_fp)

        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average MCC of {} fold：{:.4f}".format(fold_number, sum(valid_mccs) / fold_number))
    print("Average acc of {} fold：{:.4f}".format(fold_number, sum(valid_accs) / fold_number))
    print("Average spe of {} fold：{:.4f}".format(fold_number, sum(valid_spes) / fold_number))
    print("Average pre of {} fold：{:.4f}".format(fold_number, sum(valid_pres) / fold_number))
    print("Average recall of {} fold：{:.4f}".format(fold_number, sum(valid_recalls) / fold_number))
    print("Average f1 of {} fold：{:.4f}".format(fold_number, sum(valid_f1s) / fold_number))
    print("Average auc of {} fold：{:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average pr_auc of {} fold：{:.4f}".format(fold_number, sum(valid_pr_aucs) / fold_number))
    print("Average tn of {} fold：{:.4f}".format(fold_number, sum(valid_tns) / fold_number))
    print("Average tp of {} fold：{:.4f}".format(fold_number, sum(valid_tps) / fold_number))
    print("Average fn of {} fold：{:.4f}".format(fold_number, sum(valid_fns) / fold_number))
    print("Average fp of {} fold：{:.4f}".format(fold_number, sum(valid_fps) / fold_number))

    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


aver_epoch = cross_validation(dataframe, fold_number = 5)
