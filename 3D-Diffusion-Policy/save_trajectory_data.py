#!/usr/bin/env python3

import numpy as np
import os
import argparse
from diffusion_policy.dataset.xarm_2d_dataset import XArmImageDataset2D


def save_sample_trajectory(dataset, save_dir="trajectory_data", sample_idx=0):
    """
    Save a sample trajectory from the dataset.
    
    Args:
        dataset: XArmImageDataset2D instance
        save_dir: Directory to save the files
        sample_idx: Index of the sample to save
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a sample from the dataset
    sample = dataset[sample_idx]
    
    # Extract pose and action data
    pose_data = sample['obs']['pose'].numpy()  # Convert from tensor to numpy
    action_data = sample['action'].numpy()
    
    # Save the data
    pose_file = os.path.join(save_dir, "pose.npy")
    action_file = os.path.join(save_dir, "action.npy")
    
    np.save(pose_file, pose_data)
    np.save(action_file, action_data)
    
    print(f"Saved trajectory data:")
    print(f"  Pose shape: {pose_data.shape}")
    print(f"  Action shape: {action_data.shape}")
    print(f"  Files saved to: {save_dir}/")
    print(f"    - {pose_file}")
    print(f"    - {action_file}")
    
    # Print sample values
    print(f"\nSample pose (first timestep): {pose_data[0]}")
    print(f"Sample action (first timestep): {action_data[0]}")
    
    return pose_file, action_file


def main():
    parser = argparse.ArgumentParser(description='Save trajectory data from dataset')
    parser.add_argument('--zarr_path', type=str, default='/home/alex/Documents/3D-Diffusion-Policy/dt_ag/data/2d_strawberry_baseline/10_hz_baseline_100_zarr_1',
                       help='Path to the zarr dataset')
    parser.add_argument('--save_dir', type=str, default='trajectory_data',
                       help='Directory to save trajectory files')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to save')
    parser.add_argument('--horizon', type=int, default=8,
                       help='Trajectory horizon length')
    
    args = parser.parse_args()
    
    try:
        # Create dataset
        print(f"Loading dataset from: {args.zarr_path}")
        dataset = XArmImageDataset2D(
            zarr_path=args.zarr_path,
            horizon=args.horizon,
            max_train_episodes=1,  # Just load one episode for quick testing
            seed=np.random.randint(0, 1000000)
        )
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Validate sample index
        if args.sample_idx >= len(dataset):
            print(f"Warning: sample_idx {args.sample_idx} >= dataset size {len(dataset)}")
            args.sample_idx = 0
            print(f"Using sample_idx {args.sample_idx} instead")
        
        # Save trajectory data
        pose_file, action_file = save_sample_trajectory(
            dataset, args.save_dir, args.sample_idx
        )
        
        print(f"\nTo use with robot commander:")
        print(f"python robot_commander.py --pose_file {pose_file} --action_file {action_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 