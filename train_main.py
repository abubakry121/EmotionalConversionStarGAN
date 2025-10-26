"""
main.py

Author - Max Elliott

Main script to start training of the proposed model (StarGAN_emo_VC1).
"""

import argparse
import torch
import yaml
import numpy as np
import random
import os

import stargan.my_dataset as my_dataset
from stargan.my_dataset import get_filenames
import stargan.solver as solver


def make_weight_vector(filenames, data_dir):
    label_dir = os.path.join(data_dir, 'labels')
    emo_labels = []

    for f in filenames:
        label = np.load(os.path.join(label_dir, f + ".npy"))
        emo_labels.append(label[0])

    categories = list(set(emo_labels))
    total = len(emo_labels)

    counts = [total / emo_labels.count(emo) for emo in range(len(categories))]
    weights = torch.Tensor(counts)
    return weights


if __name__ == '__main__':

    # ----------------------------
    # Parse command line arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description='StarGAN-emo-VC main method')
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name for training.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Directory of checkpoint to resume training from")
    parser.add_argument("--load_emo", type=str, default=None,
                        help="Directory of pretrained emotional classifier checkpoint to use if desired.")
    parser.add_argument("--recon_only", action='store_true',
                        help='Train a model without auxiliary classifier: learn to reconstruct input.')
    parser.add_argument("--config", default='./config.yaml',
                        help='Path of config file for training.')
    parser.add_argument("-a", "--alter", action='store_true')
    args = parser.parse_args()

    # ----------------------------
    # Load YAML config safely
    # ----------------------------
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.name is not None:
        config['model']['name'] = args.name
        print(f"Model name set to: {config['model']['name']}")

    # ----------------------------
    # Set seeds for reproducibility
    # ----------------------------
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Use GPU if available
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}, Torch version: {torch.__version__}")

    # ----------------------------
    # Determine data directory
    # ----------------------------
    if config['data']['type'] == 'world':
        print("Using WORLD features.")
        assert config['model']['num_feats'] == 36
        data_dir = os.path.join(config['data']['dataset_dir'], "world")
    else:
        print("Using mel spectrograms.")
        assert config['model']['num_feats'] == 80
        data_dir = os.path.join(config['data']['dataset_dir'], "mels")

    print("Data directory =", data_dir)

    # ----------------------------
    # Make train/test split
    # ----------------------------
    files = get_filenames(data_dir)
    label_dir = os.path.join(config['data']['dataset_dir'], 'labels')
    num_emos = config['model']['num_classes']

    files = [f for f in files if np.load(os.path.join(label_dir, f + ".npy"))[0] < num_emos]
    print(len(files), "files used.")

    weight_vector = make_weight_vector(files, config['data']['dataset_dir'])
    files = my_dataset.shuffle(files)

    split_index = int(len(files) * config['data']['train_test_split'])
    train_files = files[:split_index]
    test_files = files[split_index:]

    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")

    # ----------------------------
    # Create datasets and dataloaders
    # ----------------------------
    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)
    batch_size = config['model']['batch_size']

    train_loader, test_loader = my_dataset.make_variable_dataloader(
        train_dataset, test_dataset, batch_size=batch_size
    )

    # ----------------------------
    # Initialize solver (StarGAN)
    # ----------------------------
    print("Performing whole network training.")
    s = solver.Solver(train_loader, test_loader, config,
                      load_dir=args.checkpoint, recon_only=args.recon_only)

    if args.load_emo is not None:
        s.model.load_pretrained_classifier(args.load_emo, map_location='cpu')
        print("Loaded pre-trained emotional classifier.")

    if args.alter:
        print(f"Changing loaded config to new config {args.config}.")
        s.config = config
        s.set_configuration()

    s.set_classification_weights(weight_vector)
    print("Training model.")
    s.train()
