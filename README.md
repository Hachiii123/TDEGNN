# TDEGNN: topology-enhanced self-distilled graph network for protein-DNA binding site prediction


## Abstract
Protein-DNA interactions are fundamental to diverse biological processes, and accurate identification of protein-DNA binding sites is crucial for understanding protein functions and drug target discovery. Traditional experimental approaches are time-consuming and costly, while existing computational methods remain limited in capturing global topological information and characterizing non-local contact residues. To address these challenges, we propose a topology-enhanced distillation-guided equivariant graph neural network (TDEGNN) for precise prediction of protein-DNA binding sites. The model leverages high-accuracy structures predicted by AlphaFold3, and innovatively incorporates persistent homology-based topological features fused with sequence embeddings derived from protein language models, thereby capturing both geometric and topological relationships among residues. The backbone consists of a three-layer EGNN module with identity mapping and residual connections to improve training stability. Moreover, an intra-layer self-distillation strategy is introduced to enhance model performance and generalization without additional computational cost. Results on five-fold cross-validation and independent test sets demonstrate that TDEGNN outperforms six state-of-the-art methods in terms of recall and MCC. Ablation studies further validate the effectiveness of both topological features and EGNN modules, while analysis of non-local contact residues highlights the critical role of topological features in modeling non-local dependencies within protein 3D structures.


<div align=center>
<img src="TDEGNN.jpg" width=75%>
</div>


## Preparation
### Environment Setup
```python 
   git clone https://github.com/MiJia-ID/USPDB.git
   conda env create -f environment.yml
```

## Experimental Procedure
### Create Dataset
**Firstly**, obtain the PDB files of the training and testing datasets(DNA_train_573, DNA_129_Test and DNA_181_Test ) using AlphaFold3. Details about AlphaFold3: https://github.com/google-deepmind/alphafold3?tab=readme-ov-file

Then, run the script below to create node features(PSSM,HMM,SS,AF,One-hot encoding) and distance matrix. The file can be found in scripts folder.
```python 
python3 data_io.py 
```
```python 
python3 create_adj_predict.py.py 
```

**Secondly**, run the script below to create topo features. The file can be found in feature folder.
```python 
python3 create_topo_features.py 
```

**Thirdly** , run the script below to create node features(ESM2 embeddings and ProtTrans embeddings). The file can be found in feature folder.</br>

```python 
python3 ESM2.py 
```
```python 
python3 ProtTrans.py 
```

We choose the esm2_t33_650M_UR50D pre-trained model. More details about it can be found at: https://github.com/facebookresearch/esm#available   </br>
We also choose the prot_t5_xl_uniref50 pre-trained model of ProtTrans, which uses a masked language modeling(MLM). More details about it can be found at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50    </br>

**And if you want to skip the long time preparation, We provide the pre-generated embedding files using ProtT5 and ESM2** https://drive.google.com/drive/folders/1XgDAfkKNqRZeTnSrBjEE1Gx_UdzdO-0D?usp=drive_link </br>

**Last**, run the script below to create edge features. The file can be found in feature folder.
```python 
python3 create_edge.py 
```

### Model Training
Run the following script to train the model.
```python
python3 train_val_KD.py 
```
**We also provide pre-trained models at** https://drive.google.com/drive/folders/1cI8hz59XE1iE5rQOxEvWybPSnIFnKJ1c?usp=drive_link  </br>

### Inference on Pretrained Model
Run the following script to test the model. Both test datasets, DNA_129_Test and DNA_181_Test , were included in the testing of the model.
```python
python3 test_129.py 
```
```python
python3 test_181.py 
```

