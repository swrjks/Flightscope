from generate_fighter_dataset import generate_dataset

# 12 unseen test sorties, same sampling rate as train
generate_dataset(out_dir="./data/test_basic", n_sorties=12, seed=2025, hz=10.0)
