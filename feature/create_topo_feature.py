import os
import numpy as np
import torch
import gudhi
import pandas as pd
from Bio.PDB import PDBParser,NeighborSearch
from itertools import combinations
from tqdm import tqdm  # 进度条（可选）
from sklearn.preprocessing import MinMaxScaler



class ProteinTopologyFeatureExtractor:
    def __init__(self, pdb_file=None, neighbor_distance=8.0, Cut=8):
        """
        初始化特征提取器
        
        参数:
            pdb_file: PDB文件路径
            neighbor_distance: 邻近原子搜索半径(Å)
        """
        self.pdb_file = pdb_file
        self.neighbor_distance = neighbor_distance
        self.parser = PDBParser(QUIET=True)
        if pdb_file:
            self.structure = self.parser.get_structure('protein', self.pdb_file)  # 存储结构对象
        
        # 原子类型组合配置（可根据需要修改）
        self.e_set = [
            ['C'], ['N'], ['O'],                # 单原子类型
            ['C', 'N'], ['C', 'O'], ['N', 'O'], # 两两组合
            ['C', 'N', 'O']                     # 全部三种原子
        ]

        self.Cut = Cut
        self.small = 0.01  # 条形码最小长度阈值

    def get_all_residues(self):
        """提取蛋白质中所有标准残基"""
        # structure = self.parser.get_structure('protein', self.pdb_file)
        residues = []
        
        for chain in self.structure[0]:  # 只处理第一个模型
            for residue in chain:
                if residue.id[0] == ' ':  # 只处理标准残基（排除水分子/配体）
                    # 生成标准残基标识符 c<链 id>r<残基 id>R<残基名称>
                    res_id = f"c<{chain.id}>r<{residue.id[1]}>R<{residue.resname}>"
                    # 获取CA原子坐标（若无CA则使用第一个原子）
                    if 'CA' in residue:
                        coord = residue['CA'].coord
                    else:
                        coord = next(atom for atom in residue.get_atoms()).coord
                    residues.append((res_id, coord, residue))  # 返回包含残疾信息的列表，每个元素是（res_id,coord，residue）

        # print("residues:",residues)
        print("number of residues:",len(residues))
        return residues
    
    # 以残基的 CA 原子为中心搜索邻近原子
    def find_nei_points(self, residue):
        neighbor_atoms = []
        for atom in residue.get_atoms():
            if atom.name == 'CA':
                ns = NeighborSearch(list(self.structure.get_atoms()))
                neighbors = ns.search(atom.coord, self.neighbor_distance)
                neighbor_atoms = [(a.get_name()[0], a.get_coord()) for a in neighbors]
                break
        return neighbor_atoms
    
    # 提取 0 维特征
    def _extract_h0_features(self, persistence):
        Feature_b0 = np.zeros(5)  # 初始化为全0
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) 
                            for com in persistence],
                            dtype=[('dim', int), ('birth', float), ('death', float)])
        # 只考虑死亡时间小于cutoff且长度大于small的条形码
        bars = tmpbars[(tmpbars['death'] <= self.Cut) & 
                        (tmpbars['dim'] == 0) & 
                        (tmpbars['death']-tmpbars['birth'] >= self.small)]
        
        # 计算条码的统计量特征
        if len(bars) > 0:
            lengths = bars['death'] - bars['birth']
            Feature_b0[0] = np.sum(lengths)
            Feature_b0[1] = np.min(lengths)
            Feature_b0[2] = np.max(lengths)
            Feature_b0[3] = np.mean(lengths)
            Feature_b0[4] = np.std(lengths)

        return Feature_b0

    # 提取 1 维特征
    def _extract_h1h2_features(self, persistence):
        Feature_b1 = np.zeros(15)  # 初始化为全0
        
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) 
                        for com in persistence],
                        dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[tmpbars['death'] - tmpbars['birth'] >= self.small]
        
        # 计算betti1特征，包括：条码长度、出生时间、死亡时间的统计量特征
        betti1_bars = bars[bars['dim'] == 1]
        if len(betti1_bars) > 0:
            lengths = betti1_bars['death'] - betti1_bars['birth']
            Feature_b1[0] = np.sum(lengths)
            Feature_b1[1] = np.min(lengths)
            Feature_b1[2] = np.max(lengths)
            Feature_b1[3] = np.mean(lengths)
            Feature_b1[4] = np.std(lengths)
            Feature_b1[5] = np.sum(betti1_bars['birth'])
            Feature_b1[6] = np.min(betti1_bars['birth'])
            Feature_b1[7] = np.max(betti1_bars['birth'])
            Feature_b1[8] = np.mean(betti1_bars['birth'])
            Feature_b1[9] = np.std(betti1_bars['birth'])
            Feature_b1[10] = np.sum(betti1_bars['death'])
            Feature_b1[11] = np.min(betti1_bars['death'])
            Feature_b1[12] = np.max(betti1_bars['death'])
            Feature_b1[13] = np.mean(betti1_bars['death'])
            Feature_b1[14] = np.std(betti1_bars['death'])
        return Feature_b1
    

    # 生成所有残基的拓扑特征（每残基一行
    def generate_node_features(self):
        residues = self.get_all_residues()
        fea_sum = []
        
        for res_id, coord, residue in tqdm(residues, desc="Processing residues"):
            feature_0d = []
            feature_1d = []
            neighbor_atoms = self.find_nei_points(residue)
            
            for atoms in self.e_set:
                # 筛选特定原子类型的坐标点
                points = [coord for (elem, coord) in neighbor_atoms if elem in atoms]
                
                # 计算VR复形（H0特征）
                rips = gudhi.RipsComplex(points=points)
                stree = rips.create_simplex_tree(max_dimension=1)
                h0 = self._extract_h0_features(stree.persistence())
                feature_0d.extend(h0)
                
                # 计算Alpha复形（H1/H2特征）
                alpha = gudhi.AlphaComplex(points=points)
                stree = alpha.create_simplex_tree()
                h1h2 = self._extract_h1h2_features(stree.persistence())
                feature_1d.extend(h1h2)
            
            # 合并当前残基的所有特征（7种原子组合）
            fea_sum.append(feature_0d + feature_1d)

        # 生成列名
        e_set_str = [''.join(e) for e in self.e_set]
        col_0 = [f'f0_{e}_death_{stat}' for e in e_set_str 
                for stat in ['sum', 'min', 'max', 'mean', 'std']]
        col_1 = [f'f1_{e}_{metric}_{stat}' for e in e_set_str 
                for metric in ['len', 'birth', 'death'] 
                for stat in ['sum', 'min', 'max', 'mean', 'std']]
        all_cols = col_0 + col_1


        # 创建DataFrame
        fea_df = pd.DataFrame(fea_sum, columns=all_cols)
        residue_ids = [res[0] for res in residues] 
        fea_df.insert(0, 'ID', residue_ids)

        # 归一化特征列（跳过ID列）
        scaler = MinMaxScaler()
        fea_tensor = torch.tensor(scaler.fit_transform(fea_df[all_cols]), dtype=torch.float32)  # 归一化后直接转为张量
        
        # 输出结果
        # fea_df[all_cols] = scaler.transform(fea_df[all_cols])  # 保持CSV文件也归一化
        # fea_df.to_csv(output_csv, index=False)
        
        # 返回张量+残基ID列表
        return fea_tensor, residue_ids  
    

    # 处理单个 pdb 文件
    def process_single_pdb(self, pdb_path, output_dir):
        self.pdb_file = pdb_path
        self.structure = self.parser.get_structure('protein', pdb_path)

        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        features, ids = self.generate_node_features()

        print("当前处理的蛋白质：", pdb_path)
        print()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # 保存张量
            torch.save(features, os.path.join(output_dir, f"{base_name}_topo.pt"))
            # 保存残基ID（可选）
            with open(os.path.join(output_dir, f"{base_name}_ids.txt"), 'w') as f:
                f.write('\n'.join(ids))

        return features,ids



    # 批量处理 pdb 文件
    def batch_process_pdbs(self, pdb_dir, output_dir):
        if not os.path.exists(pdb_dir):
            raise ValueError(f"PDB目录不存在: {pdb_dir}")
        os.makedirs(output_dir, exist_ok=True)

        pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                self.process_single_pdb(pdb_file, output_dir)
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
        
        print(f"所有拓扑特征已保存至 {output_dir}")

# 配置路径
#PDB_DIR = "D:/BIO-code/EGPDI/EGPDI-main/PDB"
PDB_DIR = "/home/duying/EGPDI/data/AF3PDB"
OUTPUT_DIR = "/home/duying/EGPDI/data/AF3_topo_features"

# 执行批量处理
extractor = ProteinTopologyFeatureExtractor()
extractor.batch_process_pdbs(PDB_DIR, OUTPUT_DIR)
    
    