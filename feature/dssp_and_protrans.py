import torch
from transformers import T5EncoderModel, T5Tokenizer
import re, argparse
import numpy as np
from tqdm import tqdm  # 正确导入 tqdm 函数
import multiprocessing
import os ,datetime
from Bio import pairwise2
import pickle
import gc


def get_prottrans(fasta_file, output_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1')   # 先查看当前gpu使用情况，再指定
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
        # 仅作为示例，打印前10个序列及其ID
        print(f"ID: {id}")
        print(f"sequence: {seq[:]}...")
        print("len: ",len(seq))
    
    model_path = '/home/duying/EGPDI/app/ProtTrans'
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    gpu = 2  # 使用第 3 块 GPU
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu is not None else 'cpu')
    # device = torch.device('cuda:2' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.eval().to(device)

    print(next(model.parameters()).device)
    print('starttime')
    starttime = datetime.datetime.now()
    print(starttime)
    batch_size = 1   # 每次处理 1 个序列

    for i in tqdm(range(0, len(ID_list)), batch_size):
        batch_ID_list = ID_list[i:i + batch_size]
        batch_seq_list = seq_list[i:i + batch_size]

        # 检查当前批次的所有输出文件是否已经存在
        all_files_exist = True
        for seq_id in batch_ID_list:
            out_file_path = os.path.join(output_path, seq_id + ".npy")
            if not os.path.exists(out_file_path):
                all_files_exist = False
                break
        
        # 如果当前批次所有输出文件都存在，跳过此批次
        if all_files_exist:
            print(f"批次 {i // batch_size + 1} 已处理，跳过。")
            continue

        # 处理序列:ProtTrans 不支持 U、Z、O、B，需要替换为 X（未知氨基酸）
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # 编码序列
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)  # 添加结束符；对不同长度的序列进行填充
        input_ids = torch.tensor(ids['input_ids']).to(device)  # 转换为 torch 张量
        attention_mask = torch.tensor(ids['attention_mask']).to(device)  # 指示padding位置（填充值为0）

        # 提取特征
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        print("特征尺寸大小：", embedding)

        # 保存特征
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()  # 提取非 padding 部分
            seq_emd = embedding[seq_num][:seq_len - 1]  # 去掉结束符
            print(f"蛋白质{seq_num+1}prottrans的尺寸大小: ", seq_emd.shape)  # 保存 .npy 文件

            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)



def get_dssp(fasta_file, pdb_path, dssp_path):
    DSSP = 'home/duying/EGPDI/app/DSSP'

    # 读取 .dssp 文件，提取二级结构特征 PHI,PSI,ACC,ASA,SS 
    def process_dssp(dssp_file):
        aa_type = "ACDEFGHIKLMNPQRSTVWY"  # 20 种氨基酸的字母表示
        SS_type = "HBEGITSC"              # 8 种二级结构类型
        rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                    185, 160, 145, 180, 225, 115, 140, 155, 255, 230]      # 20种氨基酸的标准溶剂可及性值
        
        with open(dssp_file, "r") as f:
            lines = f.readlines()
        
        seq = ""
        dssp_features = []

        p=0
        while lines[p].strip()[0] != "#":  # 跳过文件开头的注释行（以#开头）
            p += 1
        for i in range(p+1, len(lines)):   
            aa = lines[i][13]   # 提取氨基酸类型
            if aa == "!" or aa == "*":
                continue
            seq += aa
            SS = lines[i][16]   # 提取二级结构类型   
            if SS == " ":
                SS = "C"
            SS_vec = np.zeros(9)   # 将二级结构转换为 one-hot 形式
            SS_vec[SS_type.find(SS)] = 1
 
            PHI = float(lines[i][103:109].strip())  # 提取 PHI，PSI，ACC 值   
            PSI = float(lines[i][109:115].strip())
            ACC = float(lines[i][34:38].strip())
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100        # 计算 ASA，将其归一化到0-1之间
            dssp_features.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))  # 将 PHI,PSI,ASA 和二级结构特征拼接成一个特征向量


        return seq, dssp_features  # 返回蛋白质序列和提取到的 dssp 特征

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
    
    # 将 DSSP 特征转换为适合模型输入的形式
    def transformer_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)
        angle = dssp_feature[:, 0:2]
        ASA_SS = dssp_feature[:, 2:]

        radian = angle * (np.pi / 180)
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS], axis=1)  # 计算肽骨架扭转角 PHI 和 PSI 的正弦值和余弦值，将它们与 ASA_SS 按列拼接

        return dssp_feature
    

    # 调用 DSSP 程序生成 DSSP文件
    def get_dssp(data_path, dssp_path, ID, ref_seq):
        try:
            os.system("{} -i {}.pdb -o {}.dssp".format(DSSP, data_path + ID, dssp_path + ID))
        except Exception as e:
            print(e)
            return None

    # 读取 FASTA 文件，提取蛋白质ID和序列
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
    parser.add_argument("--fasta_file", type=str, default='/home/duying/EGPDI/data/DNA-129_Test_Extracted.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='/home/duying/EGPDI/features/prottrans_129')
    parser.add_argument("--pdb_dir", type=str, default='/home/duying/EGPDI/data/PDB')
    parser.add_argument("--dssp_output_path", type=str, default='/home/duying/EGPDI/features/dssp_129')

    args = parser.parse_args()


    # 提取 dssp 特征和 prottrans 特征
    # get_dssp(args.fasta_file, args.pdb_dir, args.dssp_output_path)
    get_prottrans(args.fasta_file, args.prottrans_output_path)


if __name__ == "__main__":
    main()