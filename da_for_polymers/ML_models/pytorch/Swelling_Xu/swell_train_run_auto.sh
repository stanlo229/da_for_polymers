model_types=('NN')
for model in "${model_types}"
do
    # SMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_SMILES --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/SMILES
    
    # AUGMENTED SMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/augmentation/StratifiedKFold/input_valid_[0-9].csv --feature_names Augmented_SMILES --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/Augmented_SMILES
    
    # BRICS
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/BRICS/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_pair_BRICS --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/BRICS
    
    # MANUAL FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/manual_frag
    
    # AUGMENTED MANUAL FRAG
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_manual_aug --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/manual_frag_aug
    
    # FINGERPRINT
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/fingerprint/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_FP_radius_3_nbits_512 --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/fingerprint
    
    # BigSMILES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_BigSMILES --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/BigSMILES
    
    # SELFIES
    python ../train.py --train_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_train_[0-9].csv --validation_path ~/Research/Repos/da_for_polymers/da_for_polymers/data/input_representation/Swelling_Xu/SMILES/StratifiedKFold/input_valid_[0-9].csv --feature_names PS_SELFIES --target_name SD --model_type "$model" --model_config ../NN/model_config.json --results_path ~/Research/Repos/da_for_polymers/da_for_polymers/training/Swelling_Xu/SELFIES --random_state 22
done