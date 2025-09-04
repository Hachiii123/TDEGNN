import torch.optim
from modules_with_edge_features import *

class MainModel(nn.Module):
    def __init__(self, dr, lr, nlayers, lamda, alpha, nfeats, num_layers=3):
        super(MainModel, self).__init__()
        
        self.num_layers = num_layers
        nhidden=512
     
        self.rgn_egnn1 = RGN_EGNN(nlayers=2, nfeat=nfeats, nhidden=nhidden, nclass=1, 
                                 dropout=dr, lamda=lamda, alpha=alpha, variant=True, heads=1)
        
       
        if num_layers >= 2:
            self.rgn_egnn2 = RGN_EGNN(nlayers=2, nfeat=nhidden, nhidden=nhidden//2, nclass=1, 
                                 dropout=dr, lamda=lamda, alpha=alpha, variant=True, heads=1)
        
        if num_layers >= 3:
            self.rgn_egnn3 = RGN_EGNN(nlayers=2, nfeat=nhidden//2, nhidden=nhidden//4, nclass=1, 
                                 dropout=dr, lamda=lamda, alpha=alpha, variant=True, heads=1)
        

        self.classifier1 = nn.Sequential(
            nn.Linear(nhidden, 128),
            nn.ReLU(),
            nn.Dropout(p=dr),
            nn.Linear(128, 1)
        )
        
        if num_layers >= 2:
            self.classifier2 = nn.Sequential(
                nn.Linear(nhidden//2, 128),
                nn.ReLU(),
                nn.Dropout(p=dr),
                nn.Linear(128, 1)
            )
            self.proj_middle = nn.Linear(256, 128)

        if num_layers >= 3:
            self.classifier3 = nn.Sequential(
                nn.Linear(nhidden//4, 128),
                nn.ReLU(),
                nn.Dropout(p=dr),
                nn.Linear(128, 1)
            )
            self.proj_shallow = nn.Linear(512, 128)
         

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-16)


    def forward(self, G, h, x, adj, efeats):

        h = torch.squeeze(h)
        x = torch.squeeze(x)
        h = h.to(torch.float32)

        fea_egnn1 = self.rgn_egnn1(G, h, x, efeats)
        fea_egnn2 = self.rgn_egnn2(G, fea_egnn1, x, efeats)
        fea_egnn3 = self.rgn_egnn3(G, fea_egnn2, x, efeats)
        
        middle_proj = self.proj_middle(fea_egnn2)
        shallow_proj = self.proj_shallow(fea_egnn1)

        logit1 = self.classifier1(fea_egnn1).squeeze(-1)  # student1
        logit2 = self.classifier2(fea_egnn2).squeeze(-1)  # student2
        logit3 = self.classifier3(fea_egnn3).squeeze(-1)  # teacher 


        return [logit3, logit2, logit1], [fea_egnn3, middle_proj, shallow_proj]
        #return [logit2, logit1], [fea_egnn2, middle_proj]

