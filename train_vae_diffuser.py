"""
TODO: fix training mixed precision -- issue with AdamW optimizer
from this issue: https://github.com/huggingface/diffusers/issues/3726
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import lpips

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(test_dataloader, vae, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    vae_model = accelerator.unwrap_model(vae)
    image_logs = []
    
    for idx, validation_image in enumerate(test_dataloader):
        images = []
        images_gt = []

        x = validation_image["pixel_values"].to(weight_dtype)
        y = validation_image["pixel_values_gt"].to(weight_dtype)

        for _ in range(1): # 1 sample

            input = x
            posterior = vae.encode(input).latent_dist
            if False:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()
            reconstructions = vae.decode(z, cross_emb=z).sample[0]

            # reconstructions = vae_model(x).sample[0]
            
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            reconstructions = (reconstructions + 1) / 2

            # convert to PIL.Image
            reconstructions = transforms.functional.to_pil_image(reconstructions)

            images.append(
                reconstructions
            )

            input_gt = y
            posterior_gt = vae.encode(input_gt).latent_dist
            z_gt = posterior_gt.mode()
            reconstructions_gt = vae.decode(z_gt, cross_emb=z).sample[0]

            reconstructions_gt = (reconstructions_gt + 1) / 2

            reconstructions_gt = transforms.functional.to_pil_image(reconstructions_gt)

            images_gt.append(
                reconstructions_gt
            )

        image_logs.append(
            {"validation_image": transforms.functional.to_pil_image(validation_image["pixel_values_normal"][0]), "images": images, "images_gt":images_gt, "idx": idx}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":

            for idx, log in enumerate(image_logs):
                images = log["images"] + log["images_gt"]
                image_idx = log["idx"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(f"{image_idx}+_{idx}", formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left) / Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.gen_images}")

    del vae_model
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a VAE training script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vae-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=4,
        help="Number of images to remove from training set to be used as validation.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--lpips_scale",
        type=float,
        default=1e-1,
        help="Scaling factor for the LPIPS metric",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load vae
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision)

    vae.decoder.requires_grad_(True)
    vae_params = vae.decoder.parameters()

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(vae_params, lr=args.learning_rate)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values_vae"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        pixel_values_normal = torch.stack([example["pixel_values_normal"] for example in examples])
        pixel_values_normal = pixel_values_normal.to(memory_format=torch.contiguous_format).float()

        pixel_values_gt = torch.stack([example["pixel_values_gt"] for example in examples])
        pixel_values_gt = pixel_values_gt.to(memory_format=torch.contiguous_format).float()

        return {
            "pixel_values": pixel_values,
            "pixel_values_normal": pixel_values_normal,
            "pixel_values_gt": pixel_values_gt,
        }

    from my_datasets.my_dataset import UnpairedLQHQDataset

    train_dataset = UnpairedLQHQDataset(
        csv_path=args.train_data_dir,  # 替换为实际路径
        tokenizer=None,
        size=args.resolution,
        placeholder_token="S",
    )

    test_dataset = UnpairedLQHQDataset(
        csv_path=args.test_data_dir,  # 替换为实际路径
        tokenizer=None,
        size=args.resolution,
        placeholder_token="S",
        max_sample=2,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        # shuffle=True, 
        collate_fn=collate_fn
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.num_train_epochs * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accelerator.init_trackers(f"{args.tracker_project_name}_{timestamp}", tracker_config)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    # if "checkpoint" in args.pretrained_model_name_or_path:
    global_step = 0
    
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        vae.decoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae.decoder):
                source = batch["pixel_values"].to(weight_dtype)
                target = batch["pixel_values_gt"].to(weight_dtype)

                # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py
                posterior = vae.encode(source).latent_dist
                z = posterior.mode()

                # middle_embedding = vae_controlnet(z, ).middle_embedding

                pred = vae.decode(z, cross_emb=z).sample

                # kl_loss = posterior.kl().mean()
                mse_loss = F.mse_loss(pred, target, reduction="mean")
                lpips_loss = lpips_loss_fn(pred, target).mean()

                loss = (
                    mse_loss + args.lpips_scale * lpips_loss #  + args.kl_scale * kl_loss
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"vae_checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        with torch.no_grad():
                            log_validation(test_dataloader, vae, accelerator, weight_dtype, global_step)

            if args.pretrained_model_name_or_path and step == 0: # if train from pretrain model
                with torch.no_grad():
                    log_validation(test_dataloader, vae, accelerator, weight_dtype, global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mse": mse_loss.detach().item(),
                "lpips": lpips_loss.detach().item(),
                "kl": kl_loss.detach().item(),
            }
            accelerator.log(logs)
            progress_bar.set_postfix(**logs)

            

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        save_path = os.path.join(
            args.output_dir, f"vae_checkpoint_final"
        )
        vae.save_pretrained(save_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()