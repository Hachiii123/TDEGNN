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
    # 设置环境变量以限制OpenMP使用的线程数
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
        line = line.strip()  # 移除行末尾的换行符和空白字符
        if line and line[0] == ">":  # 检查行是否非空且是否为ID行
            ID_list.append(line[1:])  # 去除ID行的大于号
        elif line:  # 检查行是否非空
            seq_list.append(" ".join(list(line)))  # 将非空行的序列加入序列列表中

    for id, seq in zip(ID_list[:9], seq_list[:9]):  # 仅作为示例，打印前5个序列及其ID
        print(f"ID: {id}")
        print(f"Sequence: {seq[:]}...")  # 打印序列的前50个字符（加空格后）作为示例
        print("len:",len(seq))

    model_path = "D:/BIO-code/EGPDI/EGPDI-main/app/Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()

    # 设置设备
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

        # 检查当前批次的所有输出文件是否已经存在
        all_files_exist = True
        for seq_id in batch_ID_list:
            out_file_path = os.path.join(output_path, seq_id + ".npy")
            if not os.path.exists(out_file_path):
                all_files_exist = False
                break  # 如果发现有文件不存在，则无需继续检查

        # 如果当前批次所有输出文件都存在，跳过此批次
        if all_files_exist:
            print(f"批次 {i // batch_size + 1} 已处理，跳过。")
            continue

        # 处理序列
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # 编码序列
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 提取特征
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()

        # 打印特征的尺寸大小
        print("特征尺寸大小:", embedding.shape)

        # 保存特征
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]

            # 打印特征的尺寸大小
            print(f"蛋白质特征{seq_num + 1}protrans的尺寸大小:", seq_emd.shape)

            # 打印部分内容
            # print("蛋白质特征的部分内容:")
            # print(seq_emd[:5])  # 假设您只想查看前5行的内容

            np.save(os.path.join(output_path, batch_ID_list[seq_num]), seq_emd)

    endtime = datetime.datetime.now()
    print('endtime')
    print(endtime)




def main():
    parser = argparse.ArgumentParser(description="Generate features from protein sequences.")

    parser.add_argument("--fasta_file", type=str, default='D:/BIO-code/EGPDI/EGPDI-main/DNA_Test_181_Extracted.fasta')
    parser.add_argument("--prottrans_output_path", type=str, default='D:/BIO-code/EGPDI/EGPDI-main/data/ProtTrans_181')
    parser.add_argument('--pdb_dir', type=str, default='D:/BIO-code/EGPDI/EGPDI-main/PDB/')

    args = parser.parse_args()

    # 调用之前定义的函数生成特征
    get_prottrans(args.fasta_file, args.prottrans_output_path)



if __name__ == "__main__":
    main()