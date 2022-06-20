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

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv",
)

BRICS_FRAG_DATA = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/PV_Wang/BRICS/master_brics_frag.csv"
)


class BRIC_FRAGS:
    """
    Class to fragment molecules by BRICS approach using RDKIT
    """

    def __init__(self, datapath):
        """
        Inits BRIC_FRAGS with preprocessed data

        Args:
            datapath: path to preprocessed polymer-solvent data
        """
        self.data = pd.read_csv(datapath)

    def tokenize_frag(self, brics_frag):
        """
        Function that tokenizes fragment and returns tokenization to csv file

        Args:
            brics_frag: list of all fragments for polymer-solvent pair molecules
        Returns:
            frag_dict: Dictionary of fragment
            da_pair_tokenized: Pandas series of tokenized fragments
        """
        # create dictionary of fragments
        frag_dict = {"_PAD": 0, ".": 1}
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

        # pad tokenized frags
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

    def bric_frag(self):
        """
        Fragments molecules (from SMILES) using BRICS from RDKIT

        Args:
            None

        Returns:
            Creates new master_brics_frag.csv with Labels, SMILES, DA_pairs, Fragments, PCE(%)
        """
        brics_df = pd.DataFrame(
            columns=[
                "Polymer",
                "Polymer_SMILES",
                "Solvent",
                "Solvent_SMILES",
                "Contact_angle",
                "Thickness",
                "Solvent_solubility_parameter",
                "xw_wt_percent",
                "Temperature",
                "Permeate_pressure",
                "J_total_flux",
                "a_separation_factor",
                "Polymer_BRICS",
                "Solvent_BRICS",
                "PS_pair_BRICS",
            ]
        )
        brics_df["Polymer"] = self.data["Polymer"]
        brics_df["Polymer_SMILES"] = self.data["Polymer_SMILES"]
        brics_df["Solvent"] = self.data["Solvent"]
        brics_df["Solvent_SMILES"] = self.data["Solvent_SMILES"]
        brics_df["Contact_angle"] = self.data["Contact_angle"]
        brics_df["Thickness_um"] = self.data["Thickness_um"]
        brics_df["Solvent_solubility_parameter_Mpa_sqrt"] = self.data[
            "Solvent_solubility_parameter_Mpa_sqrt"
        ]
        brics_df["xw_wt_percent"] = self.data["xw_wt_percent"]
        brics_df["Temp_C"] = self.data["Temp_C"]
        brics_df["Permeate_pressure_mbar"] = self.data["Permeate_pressure_mbar"]
        brics_df["J_Total_flux_kg_m_2h_1"] = self.data["J_Total_flux_kg_m_2h_1"]
        brics_df["a_Separation_factor_wo"] = self.data["a_Separation_factor_wo"]

        # Iterate through row and fragment using BRICS
        # to get polymer_BRICS, solvent_BRICS, and DA_pair_BRICS
        for index, row in brics_df.iterrows():
            polymer_smi = brics_df.at[index, "Polymer_SMILES"]
            solvent_smi = brics_df.at[index, "Solvent_SMILES"]
            polymer_mol = Chem.MolFromSmiles(polymer_smi)
            solvent_mol = Chem.MolFromSmiles(solvent_smi)
            polymer_brics = list(BRICS.BRICSDecompose(polymer_mol, returnMols=True))
            polymer_brics_smi = []
            for frag in polymer_brics:
                frag_smi = self.remove_dummy(frag)  # remove dummy atoms
                polymer_brics_smi.append(frag_smi)
            solvent_brics_smi = []
            solvent_brics = list(BRICS.BRICSDecompose(solvent_mol, returnMols=True))
            for frag in solvent_brics:
                frag_smi = self.remove_dummy(frag)  # remove dummy atoms
                solvent_brics_smi.append(frag_smi)

            brics_df.at[index, "Polymer_BRICS"] = polymer_brics_smi
            brics_df.at[index, "Solvent_BRICS"] = solvent_brics_smi
            # ps_pair fragments
            ps_pair = polymer_brics_smi
            ps_pair.append(".")
            ps_pair.extend(solvent_brics_smi)
            brics_df.at[index, "PS_pair_BRICS"] = ps_pair

        brics_df.to_csv(BRICS_FRAG_DATA, index=False)

    def frag_visualization(self, frag_dict):
        """
        Visualizes the dictionary of unique fragments
        NOTE: use in jupyter notebook

        Args:
            dictionary of unique fragments from polymer and solvent molecules

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
            molsPerRow=20,
            maxMols=200,
            subImgSize=(300, 300),
            legends=frag_legends,
        )
        display(img)


b_frag = BRIC_FRAGS(MASTER_MANUAL_DATA)
b_frag.bric_frag()

# print(frag_dict)
# b_frag.frag_visualization(frag_dict)
