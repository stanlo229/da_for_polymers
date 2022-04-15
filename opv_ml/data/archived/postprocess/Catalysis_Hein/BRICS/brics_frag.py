import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from collections import deque
import numpy as np
import copy
import ast

CATALYSIS_MASTER = pkg_resources.resource_filename(
    "opv_ml", "data/process/Catalysis_Hein/catalysis_master.csv"
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/Catalysis_Hein/BRICS/catalysis_brics.csv"
)


class BRIC_FRAGS:
    """
    Class to fragment molecules by BRICS approach using RDKIT
    """

    def __init__(self, datapath):
        """
        Inits BRIC_FRAGS with preprocessed data
        
        Args:
            datapath: path to preprocessed donor-acceptor data
        """
        self.data = pd.read_csv(datapath)

    def tokenize_frag(self, brics_frag):
        """
        Function that tokenizes fragment and returns tokenization to csv file

        Args:
            brics_frag: list of all fragments for donor-acceptor pair molecules
        Returns:
            frag_dict: Dictionary of fragment
            da_pair_tokenized: Pandas series of tokenized fragments
        """
        # create dictionary of fragments
        frag_dict = {
            "_PAD": 0,
        }
        for index, value in brics_frag.items():
            for frag in value:
                if frag not in frag_dict.keys():
                    frag_dict[frag] = len(frag_dict)

        print(len(frag_dict))  # 190 frags

        # tokenize frags
        da_pair_tokenized = []
        max_seq_length = 1
        for index, value in brics_frag.items():
            tokenized_list = []
            for frag in value:
                # map fragment to value in dictionary
                tokenized_list.append(frag_dict[frag])
            # for padding inputs
            if len(tokenized_list) > max_seq_length:
                max_seq_length = len(tokenized_list)
            da_pair_tokenized.append(tokenized_list)

        # pad tokenized frags from start
        for da_pair in da_pair_tokenized:
            num_of_pad = max_seq_length - len(da_pair)
            for i in range(num_of_pad):
                da_pair.insert(0, 0)

        return da_pair_tokenized, frag_dict

    def remove_dummy(self, mol):
        """
        Function that removes dummy atoms from mol and returns SMILES

        Args:
            mol: RDKiT mol object for removing dummy atoms (*)
        """
        dummy_idx = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        # remove dummy atoms altogether
        ed_mol = Chem.EditableMol(mol)
        ed_mol.BeginBatchEdit()
        for idx in dummy_idx:
            ed_mol.RemoveAtom(idx)
        ed_mol.CommitBatchEdit()
        edited_mol = ed_mol.GetMol()
        return Chem.MolToSmiles(edited_mol)

    def bric_frag(self, brics_data):
        """
        Fragments molecules (from SMILES) using BRICS from RDKIT

        Args:
            None

        Returns:
            Creates new master_brics_frag.csv with Labels, SMILES, DA_pairs, Fragments, PCE(%)
        """
        brics_df = self.data
        brics_df["Ligand_BRICS"] = ""  # create new column
        brics_df["Ligand_tokenized_BRICS"] = ""

        # Iterate through row and fragment using BRICS
        # to get ligand_BRICS, Acceptor_BRICS, and DA_pair_BRICS
        for index, row in brics_df.iterrows():
            ligand_smi = self.data.at[index, "Ligand_SMILES"]
            ligand_mol = Chem.MolFromSmiles(ligand_smi)
            ligand_brics = list(BRICS.BRICSDecompose(ligand_mol, returnMols=True))
            ligand_brics_smi = list()
            for frag in ligand_brics:
                frag_smi = self.remove_dummy(frag)  # remove dummy atoms
                ligand_brics_smi.append(frag_smi)
            brics_df.at[index, "Ligand_BRICS"] = ligand_brics_smi
        tokenized_array, frag_dict = self.tokenize_frag(brics_df["Ligand_BRICS"])
        index = 0
        for ligand in tokenized_array:
            brics_df.at[index, "Ligand_tokenized_BRICS"] = ligand
            index += 1
        brics_df.to_csv(BRICS_FRAG_DATA, index=False)

        return frag_dict

    def frag_visualization(self, frag_dict):
        """
        Visualizes the dictionary of unique fragments
        NOTE: use in jupyter notebook

        Args:
            dictionary of unique fragments from donor and acceptor molecules
        
        Returns:
            img: image of all the unique fragments
        """
        print(len(frag_dict))
        frag_list = [Chem.MolFromSmiles(frag) for frag in frag_dict.keys()]
        frag_legends = []
        for frag_key in frag_dict.keys():
            label = str(frag_dict[frag_key])
            frag_legends.append(label)

        img = Draw.MolsToGridImage(
            frag_list,
            molsPerRow=5,
            maxMols=200,
            subImgSize=(300, 300),
            legends=frag_legends,
        )
        display(img)


b_frag = BRIC_FRAGS(CATALYSIS_MASTER)
frag_dict = b_frag.bric_frag(BRICS_FRAG_DATA)
print(frag_dict)
b_frag.frag_visualization(frag_dict)
