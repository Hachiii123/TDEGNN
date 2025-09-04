import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing
import os, datetime
from Bio import pairwise2
import pickle
import argparse


def get_prottrans(fasta_file, output_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1')
    args = parser.parse_args()
    gpu = args.gpu

    ID_list = [] 
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()  
        if line and line[0] == ">":  
            ID_list.append(line[1:])  
        elif line:  
            seq_list.append(" ".join(list(line)))  

    for id, seq in zip(ID_list[:9], seq_list[:9]):  
        print(f"ID: {id}")
        print(f"Sequence: {seq[:]}...") 
        print("len:",len(seq))

    model_path = "D:/BIO-code/TDEGNN/TDEGNN-main/app/Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval().to(device)

    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1

    for i in tqdm(range(0, len(ID_list), batch_size)):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]

        all_files_exist = True
        for seq_id in batch_ID_list:
            out_file_path = os.path.join(output_path, seq_id + ".npy")
            if not os.path.exists(out_file_path):
                all_files_exist = False
                break  

        if all_files_exist:
            print(f"batch {i // batch_size + 1} is processed")
            continue

        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()


        print("embedding shape:", embedding.shape)

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]

            print(f"{seq_num + 1}:", seq_emd.shape)


            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)




def main():
    parser = argparse.ArgumentParser(description="Generate features from protein sequences.")

    parser.add_argument("--fasta_file", type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/DNA_Test_181_Extracted.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/data/ProtTrans_181')
    parser.add_argument('--pdb_dir', type=str, default='D:/BIO-code/TDEGNN/TDEGNN-main/PDB/')

    args = parser.parse_args()

    get_prottrans(args.fasta_file, args.prottrans_output_path)



if __name__ == "__main__":
    main()