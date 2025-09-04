# generate esm-2 embeddings from pre-trained protein language model
import os, argparse
import sys
import numpy as np
import torch
import esm

p = argparse.ArgumentParser()
p.add_argument('--device', type=str, default='3', help='')
p.add_argument('--output_path', type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/esm2_features', help='')
p.add_argument('--fasta_file', type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/DNA_Test_181_Extracted.fasta', help='Input fasta file')
# p.add_argument('--esm2_model', type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/app/ESM2/esm2_t48_15B_UR50D.pt', help='The path of esm2 model parameter.')
p.add_argument('--esm2_model', type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/app/ESM2_t33/esm2_t33_650M_UR50D.pt', help='The path of esm2 model parameter.')   # 内存限制，使用小的模型
args = p.parse_args()

# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model, alphabet = esm.pretrained.load_model_and_alphabet(args.esm2_model)   
batch_converter = alphabet.get_batch_converter()   
esm_model = esm_model.to(device)
esm_model.eval()


def get_esm(fasta_file, output_path):
    ID_list = []    
    seq_list = []   
    with open(fasta_file, "r")as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()    
        print(line)
        if line and line[0] == ">":    
            ID_list.append(line[1:])   
            print("ID: ",ID_list[-1])
        elif line:
            seq_list.append(" ".join(list(line)))   
            print("seq: ",line)
            batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', line)])  
            batch_tokens = batch_tokens.to(device)
        try:
            with torch.no_grad():
                results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33][0].cpu().numpy().astype(np.float16)
                esm_embed = token_representations[1:len(line) + 1]
                print("特征尺寸大小：", esm_embed.shape)
                print("len: ", len(line))
                # extracted_seq.append(line)
                np.save(os.path.join(output_path, ID_list[-1]), esm_embed)
        except Exception as e:
            print(e)


get_esm(args.fasta_file, args.output_path)

