# Self Identification (WG and SG)
model_params = {
    'model': 'pct',                # PCT
    'folder_path': "../../../../../Data/3dbsf_txt/",
    'embed_dim': 64,            # PCT: 256, ST: 64
    'num_heads': 16,
    'dropout': 0.1,           
    'ff_activation': 'gelu',
    'num_induce': 128,
    'stack': 3,
    'dropout': 0.05,             # PCT: 0.2, ST: 0.05
    'use_layernorm': False,
    'pre_layernorm': False,
    'is_final_block': False,
    'projection_dim': 128,        # PCT: 1024, SR: 128
    'pretrained_model_path': None,  # Refer execute_file
    'train_subsample_size': 8000,
    'val_subsample_size': 2048,
    'learning_rate': 1e-3        # PCT: 1e-4, ST: 1e-3
    
}