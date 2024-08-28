import os

configs = [
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.0, "sa": 0.0, "edit_threshold_c": 0.0},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.1, "sa": 0.0, "edit_threshold_c": 0.15},
    #{"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.2, "edit_threshold_c": 0.25},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.0, "edit_threshold_c": 0.30},
    #{"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.2, "edit_threshold_c": 0.35},
    #{"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.2, "edit_threshold_c": 0.40},
    #{"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.2, "edit_threshold_c": 0.45},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.3, "sa": 0.0, "edit_threshold_c": 0.5},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.4, "sa": 0.0, "edit_threshold_c": 0.75},
    #{"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.2, "sa": 0.2, "edit_threshold_c": 0.85},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.5, "sa": 0.0, "edit_threshold_c": 0.90},
    {"cfg_src": 5.0, "cfg_tar": 8.0, "num_diffusion_steps": 80, "dataset_yaml": "my.yaml", "eta": 1, "mode": "our_inv", "skip": 20, "xa": 0.6, "sa": 0.0, "edit_threshold_c": 0.95},
    # 其他配置可以添加到此列表
]

for config in configs:
    command = f"CUDA_VISIBLE_DEVICES=4 python main_run.py --cfg_src {config['cfg_src']} --cfg_tar {config['cfg_tar']} --num_diffusion_steps {config['num_diffusion_steps']} --dataset_yaml {config['dataset_yaml']} --eta {config['eta']} --mode {config['mode']} --skip {config['skip']} --xa {config['xa']} --sa {config['sa']} --edit_threshold_c {config['edit_threshold_c']}"
    
    os.system(command)