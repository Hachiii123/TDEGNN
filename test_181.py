import numpy as np
import torch
import pickle
import argparse
import warnings
import os
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix, roc_curve, precision_recall_curve, auc
from torch_geometric.utils import dense_to_sparse
import dgl
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

from model_KD import MainModel
#from model_KD_gcn import MainModel_GCN

from feature.create_node_feature_181 import create_dataset
from feature.create_graphs_181 import get_coor_test,get_adj_predicted
from feature.create_edge_181 import create_dis_matrix


warnings.filterwarnings("ignore")
seed_value = 1995
th=17

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/duying/TDEGNN/data/')
#parser.add_argument("--edgefeats_path", type=str, default='/home/duying/TDEGNN/data/AF2_Edge_feat/test_181/EdgeFeats_predicted_SC_17_181.pkl')
parser.add_argument("--edgefeats_path", type=str, default='/home/duying/TDEGNN/data/AF3_Edge_feat/test_181/EdgeFeats_predicted_SC_17_181.pkl')
parser.add_argument("--model_path", type=str, default='/home/duying/TDEGNN/models/')
args = parser.parse_args()

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


root_dir = args.dataset_path
train_path= root_dir + 'DNA-573_Train.txt'
test_path= root_dir + 'DNA_Test_181.txt'
#esm2_5120_path= root_dir + 'ESM2_t48/181/'
esm2_33_path = root_dir + 'ESM2_t33/'
ProtTrans_path = root_dir +  'ProtTrans_181/'

#AF2
#pkl_path = root_dir + 'AF2_residue_feats_181/test181/PDNA_residue_feas_PSA.pkl'
#topo_path = root_dir + 'topo_features/'
#dis_path= root_dir + 'AF2_residue_feats_181/test181/PDNA_psepos_SC.pkl'


#AF3
pkl_path= root_dir + 'AF3_residue_feats/test181/PDNA_residue_feas_PSA.pkl'
topo_path = root_dir + 'AF3_topo_features/'
dis_path= root_dir + 'AF3_residue_feats/test181/PDNA_psepos_SC.pkl'


query_ids = []
with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        if query_id[-1].islower():
            query_id += query_id[-1]
        query_ids.append(query_id)

X,y = create_dataset(query_ids,test_path,pkl_path,topo_path,esm2_33_path,ProtTrans_path,
                     residue=True,one_hot=True,esm2_33=True,prottrans=True,topo=False)
distance_matrixs=create_dis_matrix(dis_path,query_ids)


X_test = X
y_test = y

NUMBER_EPOCHS = 30
dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

IDs = query_ids
sequences = []
labels = []
with open(test_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences
labels = y_test
features = X_test

coors = get_coor_test(dis_path, query_ids)
adjs = get_adj_predicted(IDs)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

# edge features
save_edgefeats_path = args.edgefeats_path
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

test_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs,'efeats':efeats}
dataframe = pd.DataFrame(test_dic)

class dataSet(data.Dataset):
    
    def __init__(self,dataframe,adjs):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.features = dataframe['features'].values
        self.coors = dataframe['coors'].values
        self.graphs =  dataframe['graph'].values
        self.efeats = dataframe['efeats'].values
        self.adj = dataframe['adj'].values

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
    _,_,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch = map(list, zip(*samples))
    label_batch = [[label] if isinstance(label, str) else label for label in label_batch]
    graph_batch = dgl.batch(graph_batch)
    return label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch


def evaluate(model,data_loader):
    model.eval()
    epoch_loss = 0.0
    n_samples = 0

    all_true = []
    all_pred = []
    
    all_raw_features = []      
    all_latent_features = []   

    for label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:
        with torch.no_grad():
            node_features_batch = torch.tensor(node_features_batch).to(device)
            coors_batch = torch.tensor(coors_batch).to(device)
            adj_batch = adj_batch[0].to(device)
            label_batch = label_batch[0]
            efeat_batch = efeat_batch[0].to(device)
            graph_batch = graph_batch.to(device)
            
            y_true = label_batch[0]
            y_true_int = [int(label) for label in y_true]
            y_true = torch.tensor(y_true_int, dtype=torch.float32).to(device)
                                 
            # TDEGNN_full
            (outputs, _) = model(graph_batch, node_features_batch,coors_batch,adj_batch,efeat_batch)
            logits = outputs[0]

            # GCN
            #outputs = model(graph_batch, node_features_batch, coors_batch, adj_batch, efeat_batch)
            #logits = outputs['logits'][0]
            
            probs = torch.sigmoid(logits)
        
            loss = model.criterion(logits,y_true)
            epoch_loss += loss.item()
            n_samples += len(y_true)

            all_true.extend(y_true.cpu().numpy())
            all_pred.extend(probs.cpu().numpy())
            
            #all_raw_features.extend(node_features_batch.cpu().numpy())
            #all_latent_features.extend(latent_feats[-1].cpu().numpy())  

    
    
    epoch_loss_avg = epoch_loss / max(1, n_samples)

    return epoch_loss_avg, all_true, all_pred, all_raw_features, all_latent_features
    

def analysis(y_true,y_pred,best_threshold = None):
    if best_threshold == None:
        best_mcc = 0
        best_threshold = 0

        for j in range(0, 100):
            #threshold = j / 100000
            threshold = j / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

    print('best_threshold',best_threshold)
    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0

    correct_samples = sum(a == b for a, b in zip(binary_pred, y_true))
    acc = correct_samples / len(y_true)
    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    results = {
        'acc':acc,
        'spe':spe,
        'pre': pre,
        'recall': recall,
        'f1':f1,
        'mcc': mcc,
        'auc':auc,
        'aupr':aupr,
        'thred':best_threshold,
        'tn': tn,
        'tp': tp,
        'fn': fn,
        'fp': fp
    }

    return results


def test_181(Model_Path, save_features=True):
    
    # PSSM 20,Atomic features 7, SS 14 
    # one-hot 20 
    # protrans 1024 
    # topo 140
    # ESM2 1280/5120
    model = MainModel(dr,lr,nlayers,lamda,alpha,41+20+1280+1024).to(device)
    model.load_state_dict(torch.load(Model_Path,map_location=device))

    test_dataSet = dataSet(dataframe=dataframe, adjs=adjs)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=1, shuffle=False, collate_fn=graph_collate)
     
    # _, test_true, test_pred = evaluate(model, test_loader)
    _, test_true, test_pred, X_raw, X_latent = evaluate(model, test_loader)
    test_results = analysis(test_true, test_pred)
    
    if save_features:
        np.save("X_all.npy", np.array(X_raw))  
        np.save("test_layer3_features.npy", np.array(X_latent))  
        np.save("test_true_labels.npy", np.array(test_true))  

    return test_results, test_true, test_pred



# trained_model save path
Model_Path_1 = args.model_path + 'Fold1predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_2 = args.model_path + 'Fold2predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_3 = args.model_path + 'Fold3predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_4 = args.model_path + 'Fold4predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_5 = args.model_path + 'Fold5predicted_edgeFeats_best_AUPR_model.pkl'

#all_trues = []
#all_preds = []

print('begin model_fold1 prediction')
res1, true1, pred1 = test_181(Model_Path=Model_Path_1)
print('begin model_fold2 prediction')
res2, true2, pred2 = test_181(Model_Path=Model_Path_2)
print('begin model_fold3 prediction')
res3, true3, pred3 = test_181(Model_Path=Model_Path_3)
print('begin model_fold4 prediction')
res4, true4, pred4 = test_181(Model_Path=Model_Path_4)
print('begin model_fold5 prediction')
res5, true5, pred5 = test_181(Model_Path=Model_Path_5)

spe = (res1['spe']+res2['spe']+res3['spe']+res4['spe']+res5['spe'])/5
pre = (res1['pre']+res2['pre']+res3['pre']+res4['pre']+res5['pre'])/5
recall = (res1['recall']+res2['recall']+res3['recall']+res4['recall']+res5['recall'])/5
f1 = (res1['f1']+res2['f1']+res3['f1']+res4['f1']+res5['f1'])/5
mcc = (res1['mcc']+res2['mcc']+res3['mcc']+res4['mcc']+res5['mcc'])/5
auc_value = (res1['auc']+res2['auc']+res3['auc']+res4['auc']+res5['auc'])/5
aupr = (res1['aupr']+res2['aupr']+res3['aupr']+res4['aupr']+res5['aupr'])/5
avg_tn = (res1['tn'] + res2['tn'] + res3['tn'] + res4['tn'] + res5['tn']) / 5
avg_fp = (res1['fp'] + res2['fp'] + res3['fp'] + res4['fp'] + res5['fp']) / 5
avg_fn = (res1['fn'] + res2['fn'] + res3['fn'] + res4['fn'] + res5['fn']) / 5
avg_tp = (res1['tp'] + res2['tp'] + res3['tp'] + res4['tp'] + res5['tp']) / 5


print("Independent test performance on our method")
print("average spe: {:.3f} ".format(spe))
print("average pre: {:.3f}".format(pre))
print("average recall: {:.3f} ".format(recall))
print("average f1: {:.3f} ".format(f1))
print("average mcc: {:.3f} ".format(mcc))
print("average auc: {:.3f} ".format(auc_value))
print(f"True Negative (TN): {avg_tn:.1f}")
print(f"False Positive (FP): {avg_fp:.1f}")
print(f"False Negative (FN): {avg_fn:.1f}")
print(f"True Positive (TP): {avg_tp:.1f}")
print(f"AUPR: {aupr:.3f}")



all_trues = true1 + true2 + true3 + true4 + true5
all_preds = pred1 + pred2 + pred3 + pred4 + pred5


fpr, tpr, thresholds_roc = roc_curve(all_trues, all_preds)
roc_auc = auc(fpr, tpr)

roc_df = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Thresholds': thresholds_roc
})
roc_df.to_csv('181_noTopo_roc_data.csv', index=False)
print("Successfully saved in roc_curve_data.csv")


precision, recall, thresholds_pr = precision_recall_curve(all_trues, all_preds)

pr_df = pd.DataFrame({
    'Recall': recall,
    'Precision': precision,
    'Threshold': [None] + list(thresholds_pr) 
})
pr_df.to_csv('181_noTopo_pr_data.csv', index=False)
print("Successfully saved")



score_df = pd.DataFrame({
    'True_Label': all_trues,
    'Predicted_Prob': all_preds
})
score_df.to_csv('score_data.csv', index=False)
print("Successfully saved")
