from rdkit import Chem
import rdkit
from rdkit.Chem import Draw, rdchem
import pkg_resources
import pandas as pd
import ast
import copy
from collections import deque
from IPython.display import display

PV_INVENTORY = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_inventory.csv"
)

PV_EXPT_RESULT = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/PV_Wang/pv_exptresults.csv"
)

MASTER_MANUAL_DATA = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/PV_Wang/manual_frag/master_manual_frag.csv",
)

IMG_PATH = pkg_resources.resource_filename(
    "da_for_polymers", "data/input_representation/PV_Wang/manual_frag/"
)


class manual_frag:
    "Class that contains functions necessary to fragment molecules any way you want"

    def __init__(self, pv_inventory_path):
        """
        Instantiate class with appropriate data.

        Args:
            opv_data: path to ML data downloaded from Google Drive shared w/ UCSB
            donor_data: path to preprocessed donor data
            acceptor_data: path to preprocessed acceptor data

        Returns:
            None
        """
        self.pv_inventory = pd.read_csv(pv_inventory_path)

    # pipeline
    # 1 iterate with index (main)
    # 2 show molecule with atom.index
    # 3 ask for begin/end atom index OR bond index
    # 4 fragment
    # 5 show fragmented molecule
    # 6 if correct, convert to smiles and store in new .csv
    # 7 if incorrect, go back to step 3
    # 8 NOTE: be able to manually look up any donor/acceptor and re-fragment

    def lookup(self, index: int) -> str:
        """
        Function that finds and returns SMILES from donor or acceptor .csv
        
        Args:
            index: index of row in dataframe

        Returns:
            smi: SMILES of looked up molecule
        """
        try:
            smi = self.pv_inventory.at[index, "SMILES"]
        except:
            print(
                "Max index exceeded, please try again. Max index is: ",
                len(self.pv_inventory["SMILES"]) - 1,
            )

        return smi

    def fragmenter(self, smi: str):
        """
        Function that asks user how to fragment molecule

        Args:
            smi: SMILES to fragment
        
        Returns:
            ordered_frag: molecule that was fragmented by user's input, and properly ordered
        """
        # For pervaporation data, remove all dummy atoms first to keep it consistent across datasets
        # replace dummy atoms
        mol = Chem.MolFromSmiles(smi)
        edmol_frag = Chem.EditableMol(mol)
        edmol_frag.BeginBatchEdit()
        [
            edmol_frag.RemoveAtom(atom.GetIdx())
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]
        edmol_frag.CommitBatchEdit()
        mol = edmol_frag.GetMol()

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        drawn = Draw.MolToFile(mol, IMG_PATH + "manual.png", size=(700, 700))
        fragmented = False
        reformed = False
        # show all bond indexes with corresponding begin/atom idx
        for bond in mol.GetBonds():
            print(
                "bond: ",
                bond.GetIdx(),
                "begin, end: ",
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            )

        while not fragmented:
            # Ex. 30, 31, 33, 34, 101, 102
            frag_idx = input("Begin/End Atom Indexes of bond to be fragmented: ")
            if frag_idx == "None":
                mol_frag = mol
                break
            frag_tuple = tuple(map(int, frag_idx.split(", ")))
            mol_frag = Chem.FragmentOnBonds(mol, frag_tuple, addDummies=False)
            for atom in mol_frag.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            drawn = Draw.MolToFile(
                mol_frag, IMG_PATH + "manual_frag.png", size=(700, 700)
            )
            correct = input("Is the molecule fragmented correctly?: ")
            if correct == "y":
                fragmented = True

        # removes atom map numbering
        [a.SetAtomMapNum(0) for a in mol_frag.GetAtoms()]
        # replace dummy atoms
        edmol_frag = Chem.EditableMol(mol_frag)
        c_atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
        edmol_frag.BeginBatchEdit()
        [
            edmol_frag.ReplaceAtom(atom.GetIdx(), c_atom)
            for atom in mol_frag.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]
        edmol_frag.CommitBatchEdit()
        final_mol = edmol_frag.GetMol()
        drawn = Draw.MolToFile(final_mol, IMG_PATH + "manual_frag.png", size=(700, 700))
        frag_smi = Chem.MolToSmiles(final_mol)
        frag_list = frag_smi.split(".")

        # order the fragments
        frag_length = len(frag_list)
        # put placeholders
        ordered = False
        while not ordered:
            ordered_frag = []
            for i in range(frag_length):
                ordered_frag.append(i)
            for frag in frag_list:
                order_idx = int(input("Ordering of current frag (" + str(frag) + "):"))
                ordered_frag[order_idx] = frag
            print(ordered_frag)
            correct = input("Are the fragments ordered correctly?: ")
            if correct == "y":
                ordered = True

        return ordered_frag

    def return_frag_dict(self):
        """
        Sifts through manual fragments and creates unique dictionary of frag2idx

        Args:
            None
        
        Returns:
            frag_dict: dictionary of unique fragments in the combination of donor and acceptor fragmented molecules
        """
        frag_dict = {}
        frag_dict["_PAD"] = 0
        frag_dict["."] = 1
        id = len(frag_dict)
        for i in range(len(self.pv_inventory)):
            frag_str = self.pv_inventory.at[i, "Fragments"]
            frag_list = ast.literal_eval(frag_str)
            for frag in frag_list:
                if frag not in list(frag_dict.keys()):
                    frag_dict[frag] = id
                    id += 1

        return frag_dict

    def tokenize_frag(self, list_of_frag, frag_dict, max_seq_length):
        """
        Tokenizes input list of fragment from given dictionary
        * Assumes frag_dict explains all of list_of_frig

        Args:
            list_of_frag: list of all the fragments for tokenization
            frag_dict: dictionary of unique fragments from donor and acceptor molecules
            max_seq_length: the largest number of fragments for one molecule
        """
        tokenized_list = []
        # Add pre-padding
        num_of_pad = max_seq_length - len(list_of_frag)
        for i in range(num_of_pad):
            tokenized_list.append(0)

        for frag in list_of_frag:
            tokenized_list.append(frag_dict[frag])

        return tokenized_list

    def create_manual_csv(self, frag_dict, pv_expt_path, master_manual_path):
        """
        Creates master data file for manual frags

        Args:
            frag_dict: dictionary of unique fragments from donor and acceptor molecules
            pv_expt_path: path to experimental .csv for polymer swelling data
            master_manual_path: path to master .csv file for training on manual fragments
        """
        inventory_dict = {}
        for index, row in self.pv_inventory.iterrows():
            species = self.pv_inventory.at[index, "Name"]
            if species not in inventory_dict:
                inventory_dict[species] = index

        manual_df = pd.read_csv(pv_expt_path)
        manual_df["Polymer_BigSMILES"] = ""
        manual_df["Solvent_BigSMILES"] = ""
        manual_df["PS_manual"] = ""
        manual_df["PS_manual_aug"] = ""

        aug_count = 0
        # find max_seq_length
        max_seq_length = 0
        for i in range(len(manual_df)):
            polymer_label = manual_df.at[i, "Polymer"]
            mixture_label = manual_df.at[i, "Solvent"]
            polymer_frags = list(
                ast.literal_eval(
                    self.pv_inventory.at[inventory_dict[polymer_label], "Fragments"]
                )
            )
            mixture_frags = list(
                ast.literal_eval(
                    self.pv_inventory.at[inventory_dict[mixture_label], "Fragments"]
                )
            )
            max_frag_list = polymer_frags
            max_frag_list.append(".")
            max_frag_list.extend(mixture_frags)
            max_frag_length = len(max_frag_list)
            if max_frag_length > max_seq_length:
                max_seq_length = max_frag_length

        print("max_frag_length: ", max_seq_length)

        for i in range(len(manual_df)):
            polymer_label = manual_df.at[i, "Polymer"]
            mixture_label = manual_df.at[i, "Solvent"]
            polymer_frags = list(
                ast.literal_eval(
                    self.pv_inventory.at[inventory_dict[polymer_label], "Fragments"]
                )
            )
            mixture_frags = list(
                ast.literal_eval(
                    self.pv_inventory.at[inventory_dict[mixture_label], "Fragments"]
                )
            )

            # PM Pairs
            pm_pair_frags = copy.copy(polymer_frags)
            pm_pair_frags.append(".")
            pm_pair_frags.extend(mixture_frags)

            # AUGMENT Polymer (pre-ordered)
            augmented_polymer_list = []
            polymer_frag_deque = deque(copy.copy(polymer_frags))
            for j in range(len(polymer_frags)):
                frag_rotate = copy.copy(polymer_frag_deque)
                frag_rotate.rotate(j)
                frag_rotate = list(frag_rotate)
                augmented_polymer_list.append(frag_rotate)
                aug_count += 1

            # PM Pairs augmented
            pm_pair_frags_aug = []
            for aug_polymer in augmented_polymer_list:
                pm_aug_pair = copy.copy(aug_polymer)
                pm_aug_pair.append(".")
                pm_aug_pair.extend(mixture_frags)
                pm_pair_frags_aug.append(pm_aug_pair)

            # ADD TO MANUAL DF from inventory (does not separate polymer and mixture)
            manual_df.at[i, "Polymer_BigSMILES"] = self.pv_inventory.at[
                inventory_dict[polymer_label], "Polymer_BigSMILES"
            ]
            manual_df.at[i, "Solvent_BigSMILES"] = self.pv_inventory.at[
                inventory_dict[mixture_label], "Polymer_BigSMILES"
            ]
            manual_df.at[i, "PS_manual"] = pm_pair_frags
            manual_df.at[i, "PS_manual_aug"] = pm_pair_frags_aug

        # number of augmented polymers
        print("AUG POLYMERS: ", aug_count)

        manual_df.to_csv(master_manual_path, index=False)

    def bigsmiles_from_frag(self, pv_inventory_path):
        """
        Function that takes ordered fragments (manually by hand) and converts it into BigSMILES representation, specifically block copolymers
        Args:
            pv_inventory_path: path to data with manually fragmented polymers

        Returns:
            concatenates manual fragments into BigSMILES representation and returns to donor/acceptor data
        """
        # polymer/mixture BigSMILES
        self.pv_inventory["Polymer_BigSMILES"] = ""

        for index, row in self.pv_inventory.iterrows():
            big_smi = "{[][<]"
            position = 0
            if len(ast.literal_eval(self.pv_inventory["Fragments"][index])) == 1:
                big_smi = ast.literal_eval(self.pv_inventory["Fragments"][index])[0]
            else:
                for frag in ast.literal_eval(self.pv_inventory["Fragments"][index]):
                    big_smi += str(frag)
                    if (
                        position
                        == len(ast.literal_eval(self.pv_inventory["Fragments"][index]))
                        - 1
                    ):
                        big_smi += "[>][]}"
                    else:
                        big_smi += "[>][<]}{[>][<]"
                    position += 1

            self.pv_inventory["Polymer_BigSMILES"][index] = big_smi

        self.pv_inventory.to_csv(pv_inventory_path, index=False)

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
        print(frag_dict)
        frag_list = [Chem.MolFromSmiles(frag) for frag in frag_dict.keys()]
        print(frag_list)
        frag_legends = []
        for frag_key in frag_dict.keys():
            label = str(frag_dict[frag_key])
            frag_legends.append(label)

        img = Draw.MolsToGridImage(
            frag_list,
            molsPerRow=5,
            maxMols=400,
            subImgSize=(300, 300),
            legends=frag_legends,
        )
        display(img)


def cli_main():
    manual = manual_frag(PV_INVENTORY)

    # # iterate through donor and acceptor files
    # manual_df = pd.read_csv(PV_INVENTORY)
    # for i in range(16, len(manual_df["Name"])):  # len(donor_df["SMILES"])
    #     smi = manual.lookup(i)
    #     print(smi)
    #     frag_list = manual.fragmenter(smi)
    #     manual_df.at[i, "Fragments"] = frag_list
    #     manual_df.to_csv(PV_INVENTORY, index=False)

    # prepare manual frag data
    manual = manual_frag(PV_INVENTORY)
    frag_dict = manual.return_frag_dict()
    # print(len(frag_dict))
    # manual.frag_visualization(frag_dict)
    manual.bigsmiles_from_frag(PV_INVENTORY)
    manual.create_manual_csv(frag_dict, PV_EXPT_RESULT, MASTER_MANUAL_DATA)


if __name__ == "__main__":
    # cli_main()
    pass
