import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm  
import multiprocessing
import os ,datetime
from Bio import pairwise2
import pickle
import gc


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
        if line and line[0]==">":
            ID_list.append(line[1:])
        elif line:
            seq_list.append(" ".join(list(line)))
    
    for id,seq in zip(ID_list[:9], seq_list[:9]):
        print(f"ID: {id}")
        print(f"sequence: {seq[:]}...")
        print("len: ",len(seq))
    
    model_path = '/home/duying/TDEGNN/app/ProtTrans'
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    gpu = 2 
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu is not None else 'cpu')
    # device = torch.device('cuda:2' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval().to(device)

    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1   

    for i in tqdm(range(0, len(ID_list)), batch_size):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]

        all_files_exist = True
        for seq_id in batch_ID_list:
            out_file_path = os.path.join(output_path, seq_id + ".npy")
            if not os.path.exists(out_file_path):
                all_files_exist = False
                break
        
        if all_files_exist:
            print(f"batch {i // batch_size + 1} is processed, skip。")
            continue

        # Processing sequences: ProtTrans does not support U, Z, O, B, need to replace them with X (unknown amino acid)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # encoding
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)  
        input_ids = torch.tensor(ids['input_ids']).to(device) 
        attention_mask = torch.tensor(ids['attention_mask']).to(device) 
        
        # Extracting features
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # print("embedding shape:", embedding)

        # saving features
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()  # extracting non-padding portions
            seq_emd = embedding[seq_num][:seq_len - 1]  
            print(f"{seq_num+1}: ", seq_emd.shape) 

            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)



def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = 'home/duying/TDEGNN/app/DSSP'

    # reading .dssp files, extracting secondary structure features PHI, PSI, ACC, ASA, SS
    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"  
        SS_type = "HBEGITSC"            
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]      
        
        with open(dssp_file, "r") as f:
            lines = f.readlines()
        
        seq = ""
        dssp_features = []

        p=0
        while lines[p].strip()[0] != "#":  
            p += 1
        for i in range(p+1, len(lines)):   
            aa = lines[i][13]  
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]    
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9)  
            SS_vec[SS_type.find(SS)] = 1  
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100        
            dssp_features.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))  


        return seq, dssp_features 

    # 
    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)
        ref_seq = alignments[0].seqA
        seq = alignments[0].seqB

        SS_vec = np.zeros(9)
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array[360, 360, 0]), SS_vec)

        new_dssp = []
        for aa in seq:
            if aa == "-":
                new_dssp.append(padded_item)
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-":
                continue
            matched_dssp.append(new_dssp[i])
        
        return matched_dssp
    
    def transformer_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:, 0:2]
        ASA_SS = dssp_feature[:, 2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)  

        return dssp_feature
 

    # generating dssp files
    def get_dssp(data_path, dssp_path, ID, ref_seq):
        try:
            os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID))
        except Exception as e:
            print(e)
            return None

    # reading fasta files
    pdbfasta = {}
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()
        for i in range(len(fasta_ori)):
            if fasta_ori[i][0] == ">":
                name = fasta_ori[i].split('>')[1].replace('\n', '')
                seq = fasta_ori[i+1].replace('\n', '')
                pdbfasta[name] = seq
        
        fault_name = []
        for name in pdbfasta.keys():
            sign = get_dssp(pdb_path, dssp_path, name, pdbfasta[name])
            if sign == None:
                fault_name.append(name)
        



def main():
    parser = argparse.ArgumentParser(description="Generate features from protein sequences.")
    parser.add_argument("--fasta_file", type=str, default='/home/duying/TDEGNN/data/DNA-129_Test_Extracted.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='/home/duying/TDEGNN/features/prottrans_129')
    parser.add_argument("--pdb_dir", type=str, default='/home/duying/TDEGNN/data/PDB')
    parser.add_argument("--dssp_output_path", type=str, default='/home/duying/TDEGNN/features/dssp_129')

    args = parser.parse_args()


    # extracting dssp and protTrans features
    # get_dssp(args.fasta_file, args.pdb_dir, args.dssp_output_path)
    get_prottrans(args.fasta_file, args.prottrans_output_path)


if __name__ == "__main__":
    main()