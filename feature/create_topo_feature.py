import os
import numpy as np
import torch
import gudhi
import pandas as pd
from Bio.PDB import PDBParser,NeighborSearch
from itertools import combinations
from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler



class ProteinTopologyFeatureExtractor:
    def __init__(self, pdb_file=None, neighbor_distance=8.0, Cut=8):
        """
        Initialize the feature extractor

        Parameters:
            pdb_file: Path to the PDB file
            neighbor_distance: Radius for searching neighboring atoms (Å)
        """
        self.pdb_file = pdb_file
        self.neighbor_distance = neighbor_distance
        self.parser = PDBParser(QUIET=True)
        if pdb_file:
            self.structure = self.parser.get_structure('protein', self.pdb_file)  # 存储结构对象
        
        # Atom type combination configuration
        self.e_set = [
            ['C'], ['N'], ['O'],                
            ['C', 'N'], ['C', 'O'], ['N', 'O'],
            ['C', 'N', 'O']                     
        ]

        self.Cut = Cut
        self.small = 0.01  # Minimum length threshold for barcodes

    def get_all_residues(self):
        """Extract all standard residues in the protein"""
        residues = []
        
        for chain in self.structure[0]:  
            for residue in chain:
                if residue.id[0] == ' ': 
                    res_id = f"c<{chain.id}>r<{residue.id[1]}>R<{residue.resname}>"
                    if 'CA' in residue:
                        coord = residue['CA'].coord
                    else:
                        coord = next(atom for atom in residue.get_atoms()).coord
                    residues.append((res_id, coord, residue))  

        # print("residues:",residues)
        print("number of residues:",len(residues))
        return residues
    
    # Search for neighboring atoms centered on the residue's CA atom
    def find_nei_points(self, residue):
        neighbor_atoms = []
        for atom in residue.get_atoms():
            if atom.name == 'CA':
                ns = NeighborSearch(list(self.structure.get_atoms()))
                neighbors = ns.search(atom.coord, self.neighbor_distance)
                neighbor_atoms = [(a.get_name()[0], a.get_coord()) for a in neighbors]
                break
        return neighbor_atoms
    
    # Extract 0-dimensional features
    def _extract_h0_features(self, persistence):
        Feature_b0 = np.zeros(5)  
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) 
                            for com in persistence],
                            dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[(tmpbars['death'] <= self.Cut) & 
                        (tmpbars['dim'] == 0) & 
                        (tmpbars['death']-tmpbars['birth'] >= self.small)]
        
       
        if len(bars) > 0:
            lengths = bars['death'] - bars['birth']
            Feature_b0[0] = np.sum(lengths)
            Feature_b0[1] = np.min(lengths)
            Feature_b0[2] = np.max(lengths)
            Feature_b0[3] = np.mean(lengths)
            Feature_b0[4] = np.std(lengths)

        return Feature_b0

    # # Extract 1-dimensional features
    def _extract_h1h2_features(self, persistence):
        Feature_b1 = np.zeros(15)  
        
        tmpbars = np.array([(int(com[0]), float(com[1][0]), float(com[1][1])) 
                        for com in persistence],
                        dtype=[('dim', int), ('birth', float), ('death', float)])
        bars = tmpbars[tmpbars['death'] - tmpbars['birth'] >= self.small]
        
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
    

    # Generate topological features for all residues (one row per residue)
    def generate_node_features(self):
        residues = self.get_all_residues()
        fea_sum = []
        
        for res_id, coord, residue in tqdm(residues, desc="Processing residues"):
            feature_0d = []
            feature_1d = []
            neighbor_atoms = self.find_nei_points(residue)
            
            for atoms in self.e_set:
                points = [coord for (elem, coord) in neighbor_atoms if elem in atoms]
                
                # VR complex
                rips = gudhi.RipsComplex(points=points)
                stree = rips.create_simplex_tree(max_dimension=1)
                h0 = self._extract_h0_features(stree.persistence())
                feature_0d.extend(h0)
                
                # Alpha complex
                alpha = gudhi.AlphaComplex(points=points)
                stree = alpha.create_simplex_tree()
                h1h2 = self._extract_h1h2_features(stree.persistence())
                feature_1d.extend(h1h2)
            
            fea_sum.append(feature_0d + feature_1d)


        e_set_str = [''.join(e) for e in self.e_set]
        col_0 = [f'f0_{e}_death_{stat}' for e in e_set_str 
                for stat in ['sum', 'min', 'max', 'mean', 'std']]
        col_1 = [f'f1_{e}_{metric}_{stat}' for e in e_set_str 
                for metric in ['len', 'birth', 'death'] 
                for stat in ['sum', 'min', 'max', 'mean', 'std']]
        all_cols = col_0 + col_1


        fea_df = pd.DataFrame(fea_sum, columns=all_cols)
        residue_ids = [res[0] for res in residues] 
        fea_df.insert(0, 'ID', residue_ids)

        scaler = MinMaxScaler()
        fea_tensor = torch.tensor(scaler.fit_transform(fea_df[all_cols]), dtype=torch.float32)  # 归一化后直接转为张量

        return fea_tensor, residue_ids  
    

    def process_single_pdb(self, pdb_path, output_dir):
        self.pdb_file = pdb_path
        self.structure = self.parser.get_structure('protein', pdb_path)

        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        features, ids = self.generate_node_features()

        print("current protein:", pdb_path)
        print()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(features, os.path.join(output_dir, f"{base_name}_topo.pt"))
            with open(os.path.join(output_dir, f"{base_name}_ids.txt"), 'w') as f:
                f.write('\n'.join(ids))

        return features,ids


    def batch_process_pdbs(self, pdb_dir, output_dir):
        if not os.path.exists(pdb_dir):
            raise ValueError(f"PDB dir does not exist: {pdb_dir}")
        os.makedirs(output_dir, exist_ok=True)

        pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                self.process_single_pdb(pdb_file, output_dir)
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
        
        print(f"All protein's topo features saved in {output_dir}")

#PDB_DIR = "D:/BIO-code/TDEGNN/TDEGNN-main/PDB"
PDB_DIR = "/home/duying/TDEGNN/data/AF3PDB"
OUTPUT_DIR = "/home/duying/TDEGNN/data/AF3_topo_features"

extractor = ProteinTopologyFeatureExtractor()
extractor.batch_process_pdbs(PDB_DIR, OUTPUT_DIR)
    
    