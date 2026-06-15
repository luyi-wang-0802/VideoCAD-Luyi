"""
Main training + evaluation script.
"""
import warnings
# Filter out Matplotlib 3D projection warning
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
# Filter out PyTorch deprecated warning
warnings.filterwarnings("ignore", message="'has_cuda' is deprecated")

import argparse
from data_loader.data_loader import create_dataloader, create_dataset_from_config
from data_loader.device_cached_loader import DeviceCachedDataLoader
from model.model_factory import ModelFactory, ModelType
from trainer import create_trainer
import torch
from experiment import Experiment, resolve_config_entry
import utils
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
import time


# Enable TF32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def output_subdir(output_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.join(output_dir, path)


def args_to_dict(args):
    return {key: value for key, value in vars(args).items()}


def cache_packet_on_device(packet, device, split_name):
    print(f"Caching {split_name} batches on {device}...")
    cached_loader = DeviceCachedDataLoader.from_loader(
        packet["loader"],
        device=device,
        desc=f"Caching {split_name} on {device}",
    )
    print(f"Cached {len(cached_loader)} {split_name} batches on {device}")
    return {"loader": cached_loader, "sampler": None}


def main(rank=None, world_size=None, gpu_ids=None, args=None): 
    if args is None: # hack to allow for multi gpu training
        raise ValueError("args must be provided")
    experiment = None

    if rank is None or world_size is None:
        # Single GPU or CPU mode
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        distributed = False
    else:
        # Distributed training mode
        if gpu_ids is not None:
            # Set CUDA_VISIBLE_DEVICES to only show the GPU for this process
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[rank])
            # Always use cuda:0 since we've set CUDA_VISIBLE_DEVICES
            device = torch.device('cuda:0')
            # Set the current device
            torch.cuda.set_device(device)
        else:
            device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(device)
        
        setup(rank, world_size)
        distributed = True

    # Set multiprocessing start method to 'spawn' to avoid issues
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method('spawn')

    model_params = utils.load_json(args.model_config)
    config_name, selected_model_params = resolve_config_entry(model_params, args.model_name)
    selected_model_params["use_gencad_augmentation"] = args.enable_random
    selected_model_params["dataset_path"] = args.dataset_path
    model_params[config_name] = selected_model_params
    num_views = selected_model_params.get("num_views", 0)
    assert num_views == len(args.view_ids) or num_views == 0
    if num_views == 0:
        args.view_ids = []
    output_dir = args.output_dir
    training_runs_dir = output_subdir(output_dir, args.training_runs_dir)
    if rank in {None, 0}:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(training_runs_dir, exist_ok=True)

    training_config = {
        'dataset_path': args.dataset_path,
        'batch_size': args.batch_size,
        'lr': 1e-5,
        'override_lr': args.lr,
        'num_workers': args.num_workers,
        'save_frequency': 20,
        'output_dir': output_dir,
        'training_runs_dir': training_runs_dir,
        'model_config_path': args.model_config,
        'dataset_summary_path': os.path.join(args.dataset_path, "summary.json"),
        'command_args': args_to_dict(args),
        'val_frequency': 4,
        'compile': args.compile, #VPT cannot be compiled
        'enable_random': selected_model_params.get("enable_random", True),
        'sequential': False,
        'seq_val_frequency': 1100,
        'epochs': args.epochs,
        'enable_parallel': args.enable_parallel,
        'early_stopping_enabled': True,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'early_stopping_metric': 'loss',
        'early_stopping_mode': 'min',
        'use_mse': True,
        'rank': rank if distributed else 0,
        'world_size': world_size if distributed else 1,
        'gpu_ids': gpu_ids if distributed else [0],  # Add GPU IDs to config
        'persistent_workers': False,  # Disable persistent workers to avoid the error
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_run_name': args.wandb_run_name,
        'load_global_floorplan': args.load_global_floorplan,
        'image_dtype': args.image_dtype,
        'command_line_overrides': {
            'early_stopping_enabled': args.early_stopping_enabled,
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_min_delta': args.early_stopping_min_delta,
            'early_stopping_metric': args.early_stopping_metric,
            'early_stopping_mode': args.early_stopping_mode,
            'val_frequency': args.val_frequency,
        },
    }

    frame_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image_transform = transforms.Normalize(mean=[0.5], std=[0.5])
    
    train_packet, val_packet, test_packet = create_dataset_from_config(
        dataset_path=args.dataset_path, 
        config=args.config_path, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        multiview_dir=args.multiview_dir,
        view_ids=args.view_ids,
        frame_transform=frame_transform,
        image_transform=image_transform,
        distributed=distributed,
        rank=rank if distributed else 0,
        world_size=world_size if distributed else 1,
        image_dir=args.image_dir,
        enable_random=args.enable_random,
        sequence_retriever=args.sequence_retriever,
        overfit=args.overfit_data,
        load_global_floorplan=args.load_global_floorplan,
        image_dtype=args.image_dtype,
    )
    if args.cache_dataset_on_device:
        train_packet = cache_packet_on_device(train_packet, device, "train")
        val_packet = cache_packet_on_device(val_packet, device, "val")
        test_packet = cache_packet_on_device(test_packet, device, "test")
    
    experiment = Experiment(
        train_packet=train_packet,
        val_packet=val_packet,
        test_packet=test_packet,
        device=device,
        num_workers=args.num_workers,
        training_config=training_config,
        rank=rank if distributed else 0
    )

    try:
        start_time = time.time()
        experiment.run_experiment_with_config(model_params, args.model_name)
        end_time = time.time()
        print(f"Total training time: {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    finally:
        if experiment is not None:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass
        # Always clean up
        if dist.is_initialized():
            dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--dataset_path", type=str, default="processed_data")
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--enable_random", type=str2bool, default=True)
    parser.add_argument("--image_dir", type=str, default="data/data_raw/images")
    parser.add_argument("--sequence_retriever", type=str, default="optimized")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--training_runs_dir", type=str, default="training_runs")
    parser.add_argument("--multiview_dir", type=str, default="multi_view_images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--view_ids", type=list, default=["05", "09", "20"])
    parser.add_argument("--model_config", type=str, default="model_configs/primitive_action_policy.json")
    parser.add_argument("--model_name", type=str, default="primitive_action_policy")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None, help="Override the initial learning rate from model config.")
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="videocad-primitive-action")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--enable_parallel", type=str2bool, default=True)
    parser.add_argument("--overfit_data", type=str2bool, default=False)
    parser.add_argument("--cache_dataset_on_device", type=str2bool, default=False)
    parser.add_argument("--load_global_floorplan", type=str2bool, default=False)
    parser.add_argument("--image_dtype", type=str, choices=["float32", "float16"], default="float32")
    parser.add_argument("--early_stopping_enabled", type=str2bool, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--early_stopping_metric", type=str, default=None)
    parser.add_argument("--early_stopping_mode", type=str, choices=["min", "max"], default=None)
    parser.add_argument("--val_frequency", type=int, default=None)
    args = parser.parse_args()


    if args.gpu_ids:
        # Parse GPU IDs and set CUDA_VISIBLE_DEVICES
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        world_size = len(gpu_ids)
    else:
        # Use all available GPUs
        gpu_ids = list(range(torch.cuda.device_count()))
        world_size = len(gpu_ids)

    if world_size == 0:
        print("No GPUs available. Using CPU.")
        main(args=args)
    elif world_size == 1:
        print(f"Using single GPU: {gpu_ids[0]}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        main(args=args)
    else:
        print(f"Using {world_size} GPUs: {gpu_ids}")
        mp.spawn(main, args=(world_size, gpu_ids, args), nprocs=world_size, join=True)
