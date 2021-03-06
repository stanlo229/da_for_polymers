model_types=('RF' 'BRT' 'SVM')
for model in "${model_types}"
do
    # AUGMENTED SMILES
    python ../train.py --train_path ../../../data/input_representation/PV_Wang/augmentation/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/augmentation/StratifiedKFold/input_valid_[0-9].csv --feature_names Augmented_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/BRICS/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/BRICS/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_pair_BRICS,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/fingerprint/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/fingerprint/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_FP_radius_3_nbits_512,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/fingerprint --random_state 22
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/manual_frag --random_state 22
    
    # AUGMENTED MANUAL FRAG
    python ../train.py --train_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual_aug,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/manual_frag_aug --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_SMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_SELFIES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/PV_Wang/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_BigSMILES,Contact_angle,Thickness_um,Solvent_solubility_parameter_Mpa_sqrt,xw_wt_percent,Temp_C,Permeate_pressure_mbar --target_name J_Total_flux_kg_m_2h_1 --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./pv_hpo_space.json --results_path ../../../training/PV_Wang/BigSMILES --random_state 22
done