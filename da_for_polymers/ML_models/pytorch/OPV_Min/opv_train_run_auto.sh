model_types=('NN')
for model in "${model_types}"
do
    # SMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_SMILES --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/SMILES
    
    # AUGMENTED SMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/augmentation/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/augmentation/KFold/input_valid_[0-9].csv --feature_names Augmented_SMILES --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/Augmented_SMILES
    
    # BRICS
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/BRICS/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/BRICS/KFold/input_valid_[0-9].csv --feature_names DA_pair_BRICS --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/BRICS
    
    # MANUAL FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/manual_frag/KFold/input_valid_[0-9].csv --feature_names DA_manual --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/manual_frag
    
    # AUGMENTED MANUAL FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/manual_frag/KFold/input_valid_[0-9].csv --feature_names DA_manual_aug --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/manual_frag_aug
    
    # FINGERPRINT
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/fingerprint/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/fingerprint/KFold/input_valid_[0-9].csv --feature_names DA_FP_radius_3_nbits_512 --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/fingerprint
    
    # BigSMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_BigSMILES --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/BigSMILES
    
    # SELFIES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_SELFIES --target_name calc_PCE_percent --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/OPV_Min/SELFIES --random_state 22
done