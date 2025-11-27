import os
import time
import random
from argparse import Namespace

import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from args import get_parser, process_args
from utils import printt, get_suffix
from sea.model import load_model
from sea.data import DataModule


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _build_args_from_params(params):
    """
    Build the training args starting from the CLI defaults, then
    overriding with the provided parameter dict / Namespace.

    This lets you specify everything (data + training hyperparameters)
    in a single ``sea_data_module_params`` style mapping.
    """
    parser = get_parser()
    # Start from parser defaults (no CLI)
    args = parser.parse_args([])

    if isinstance(params, Namespace):
        items = vars(params).items()
    else:
        items = dict(params).items()

    for k, v in items:
        setattr(args, k, v)

    # Derive args_file / results_file and apply config file overrides
    process_args(args)
    return args


def train_sea(
    train_batched_data,
    test_batched_data,
    train_batched_graphs,
    test_batched_graphs,
    sea_data_module_params,
    device = "cuda",
):
    """
    Train the SEA model given batched node features/graphs and a parameter dict.

    Parameters
    ----------
    train_batched_data, test_batched_data:
        Lists/arrays of node feature batches, e.g. output of your
        ``prepare_node_features`` + ``split_data`` logic.
    train_batched_graphs, test_batched_graphs:
        Lists/arrays of adjacency matrices (one per graph) split into
        train and test.
    sea_data_module_params:
        A ``dict`` or ``argparse.Namespace`` containing ALL SEA arguments
        you care about (data, sampler, model, and training), e.g. your
        ``sea_data_module_params`` from the Snakemake script. Anything
        not specified here falls back to the original CLI defaults.

    Returns
    -------
    model:
        The trained SEA LightningModule, loaded from the best checkpoint
        (if checkpointing is enabled).
    """
    printt("Starting SEA training...")

    # Build args object compatible with the original training script
    args = _build_args_from_params(sea_data_module_params)

    # Construct DataModule from the provided batched data/graphs
    data_module = DataModule(
        train_batched_data=train_batched_data,
        test_batched_data=test_batched_data,
        train_batched_graphs=train_batched_graphs,
        test_batched_graphs=test_batched_graphs,
        args=args,
    )

    # Ensure num_vars matches the underlying dataset if available
    try:
        sample_ds = data_module.subset_train.data[0]
        if hasattr(sample_ds, "num_vars"):
            args.num_vars = sample_ds.num_vars
    except Exception:
        pass

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # Save args (for reproducibility; human-readable YAML)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    with open(args.args_file, "w+") as f:
        yaml.dump(vars(args), f)

    # Setup model and randomness
    _set_seed(args.seed)
    model = load_model(args)
    printt("Finished loading model.")

    # Logger
    if args.debug:
        wandb_logger = None
    else:
        run_id = str(time.time())
        wandb_logger = WandbLogger(
            project=args.project_name,
            entity=getattr(args, "entity", None),
            name=args.run_name,
        )
        wandb_logger.watch(model)
        # Save checkpoints/run-specific files under a subdirectory
        args.save_path = os.path.join(args.save_path, run_id)
        os.makedirs(args.save_path, exist_ok=True)

    # Determine whether to maximize or minimize the monitored metric
    mode = "max"
    for keyword in ["loss"]:
        if keyword in args.metric:
            mode = "min"

    checkpoint_kwargs = {
        "save_top_k": 1,
        "monitor": args.metric,
        "mode": mode,
        "filename": get_suffix(args.metric),
        "dirpath": args.save_path,
        "save_last": True,
    }
    cb_checkpoint = ModelCheckpoint(**checkpoint_kwargs)

    cb_earlystop = EarlyStopping(
        monitor=args.metric,
        patience=args.patience,
        mode=mode,
    )
    cb_lr = LearningRateMonitor(logging_interval="step")

    callbacks = [
        RichProgressBar(),
        cb_checkpoint,
        cb_earlystop,
        # cb_lr  # enable if you want LR logging
    ]
    if args.no_tqdm:
        callbacks[0].disable()

    device_ids = [gpu + i for i in range(num_gpu)]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "accumulate_grad_batches": args.accumulate_batches,
        "gradient_clip_val": 1.0,
        # evaluate more frequently (can be overridden via override_kwargs if needed)
        "limit_train_batches": 200,
        "limit_val_batches": 50,
        # logging and saving
        "callbacks": callbacks,
        "log_every_n_steps": args.log_frequency,
        "fast_dev_run": args.debug,
        "logger": wandb_logger,
        # GPU utilization
        "devices": device_ids,
        "accelerator": "gpu",
        "strategy": "ddp",
        # "precision": 16,  # mixed precision not enabled by default
    }

    trainer = pl.Trainer(**trainer_kwargs)
    printt("Initialized trainer.")

    # Resume training from checkpoint if requested
    fit_kwargs = {}
    if getattr(args, "checkpoint_path", "") and os.path.exists(args.checkpoint_path):
        fit_kwargs["ckpt_path"] = args.checkpoint_path

    # Train
    trainer.fit(model, data_module, **fit_kwargs)

    # Find best checkpoint
    if not args.debug:
        best_path = cb_checkpoint.best_model_path
        printt(f"Best model checkpoint: {best_path}")
    else:
        best_path = None

    # If we have a best checkpoint, load it before returning.
    if best_path is not None:
        model = model.load_from_checkpoint(best_path)

    # Freeze model and return for downstream pipelines (e.g., Snakemake)
    model.eval()
    printt("SEA training complete.")

    return model


def load_sea_model(model_weights_path, sea_data_module_args):
    """
    Load a trained SEA model from a checkpoint, given the arguments that were
    used during training.

    Parameters
    ----------
    model_weights_path:
        Path to the Lightning checkpoint produced by ``train_sea`` (or the
        original training script).
    sea_data_module_args:
        A ``dict`` or ``argparse.Namespace`` containing the SEA arguments
        that were used for training and pickled to disk. This should be the
        same object you pass / saved as ``sea_data_module_args`` in your
        Snakemake pipeline.

    Returns
    -------
    model:
        The SEA LightningModule restored from the checkpoint, in eval mode.
    """
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_weights_path}")

    # Rebuild args in the same way as for training, so we know which model
    # class to instantiate (aggregator vs baseline, etc.).
    args = _build_args_from_params(sea_data_module_args)

    base_model = load_model(args)
    model = base_model.load_from_checkpoint(model_weights_path)
    model.eval()
    return model


def run_sea_on_test(
    test_batched_data,
    test_batched_graphs,
    sea_data_module_params,
    model,
    device: str = "cuda",
):
    """
    Run a trained SEA model on *only* the test batched data/graphs.

    This builds a temporary ``DataModule`` whose predict dataloader is backed
    by ``test_batched_*`` and uses PyTorch Lightning's ``Trainer.predict``
    to obtain the model outputs.

    Parameters
    ----------
    test_batched_data, test_batched_graphs:
        Lists/arrays of node features and adjacency matrices for the test set.
    sea_data_module_params:
        Same params object (dict or Namespace) you used for training /
        constructing ``sea_data_module_args``.
    model:
        A trained SEA model (e.g., from ``train_sea`` or ``load_sea_model``).
    device:
        "cuda" / "gpu" (default) or "cpu".

    Returns
    -------
    results:
        A dict with aggregated prediction outputs over the test set. The
        ``"pred"`` entries correspond to the per-edge predictions returned
        by the SEA model (same format as in the original ``inference.py``).
    """
    # Rebuild args so that DataModule behavior (samplers, batch sizes, etc.)
    # matches what was used during training.
    args = _build_args_from_params(sea_data_module_params)

    # Dummy/empty train split; we only care about subset_test / predict_dataloader
    data_module = DataModule(
        train_batched_data=[],
        test_batched_data=test_batched_data,
        train_batched_graphs=[],
        test_batched_graphs=test_batched_graphs,
        args=args,
    )

    # Configure accelerator/devices
    if device in ["cuda", "gpu"]:
        accelerator = "gpu"
        devices = [getattr(args, "gpu", 0)]
    else:
        accelerator = "cpu"
        devices = None

    trainer_kwargs = {
        "accelerator": accelerator,
        "enable_checkpointing": False,
        "logger": False,
    }
    if devices is not None:
        trainer_kwargs["devices"] = devices

    tester = pl.Trainer(**trainer_kwargs)
    outputs = tester.predict(model, data_module)

    # Aggregate over test batches (mirror style of ``inference.py``)
    from collections import defaultdict as _dd

    results_dict = _dd(list)
    for batch in outputs:
        for k, v in batch.items():
            if isinstance(v, list):
                results_dict[k].extend(v)
            else:
                results_dict[k].append(v)

    return dict(results_dict)
