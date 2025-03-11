import os
import argparse
import torch
import glob
import re
from datetime import datetime


def find_best_model(pattern="models/*_best.pth"):
    """Find the best model based on score in filename or metadata"""
    best_models = glob.glob(pattern)

    if not best_models:
        print("No best models found.")
        return None

    best_score = -float('inf')
    best_model_path = None

    for model_path in best_models:
        try:
            # Load model to get stored best score
            checkpoint = torch.load(
                model_path, map_location=torch.device('cpu'))
            if 'best_score' in checkpoint:
                score = checkpoint['best_score']
                print(f"Model: {model_path}, Score: {score}")

                if score > best_score:
                    best_score = score
                    best_model_path = model_path
        except Exception as e:
            print(f"Error loading {model_path}: {e}")

    if best_model_path:
        print(f"Selected best model: {best_model_path} (Score: {best_score})")

    return best_model_path


def setup_next_iteration(args):
    """Set up the next training iteration using the best model from previous runs"""
    # Find the best model
    best_model_path = args.model_path
    if not best_model_path:
        best_model_path = find_best_model()
        if not best_model_path:
            print("No best model found. Will start fresh training.")
            return None

    if not os.path.exists(best_model_path):
        print(f"Error: Model file not found: {best_model_path}")
        return None

    # Create new run name if not provided
    run_name = args.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iteration_num = args.iteration
        run_name = f"mario_iter{iteration_num}_{timestamp}"

    # Load the checkpoint
    try:
        checkpoint = torch.load(
            best_model_path, map_location=torch.device('cpu'))
        print(f"Successfully loaded model from {best_model_path}")

        # Create a new checkpoint for the next iteration
        checkpoint_path = f"checkpoints/{run_name}_initial.pth"

        # If we need to reset certain aspects
        if args.reset_optimizer:
            if 'optimizer_state_dict' in checkpoint:
                del checkpoint['optimizer_state_dict']

        if args.reset_steps:
            checkpoint['steps_done'] = 0
            checkpoint['episode'] = 0

        # Save the new checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(
            f"Created new checkpoint for next iteration at {checkpoint_path}")

        # Generate command for next training run
        cmd = f"python progressive_mario_trainer.py --run-name {run_name} --load-checkpoint {run_name}_initial.pth"

        if args.custom_params:
            cmd += f" {args.custom_params}"

        print("\nTo start the next iteration, run:")
        print(cmd)

        return checkpoint_path

    except Exception as e:
        print(f"Error processing checkpoint: {e}")
        return None


def main(args):
    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Setup next iteration
    setup_next_iteration(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up next training iteration with best model")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to specific model to use (if not provided, will find best)")
    parser.add_argument("--run-name", type=str, default="",
                        help="Name for the next training run")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Reset optimizer state")
    parser.add_argument("--reset-steps", action="store_true",
                        help="Reset steps and episode counter")
    parser.add_argument("--custom-params", type=str, default="",
                        help="Additional parameters to pass to the training script")

    args = parser.parse_args()
    main(args)
