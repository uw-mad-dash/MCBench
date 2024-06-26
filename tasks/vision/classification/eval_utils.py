# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation utilities."""

import os
from functools import partial

import torch

from megatron import get_args
from megatron import print_rank_0, print_rank_last
from megatron import mpu
from megatron.schedules import get_forward_backward_func
from tasks.vision.finetune_utils import build_data_loader
from tasks.vision.finetune_utils import process_batch
from torchvision import datasets, transforms

from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
    ConvertImageDtype,
)


def accuracy_func_provider_vit():
    """Provide function that calculates accuracies."""
    args = get_args()
    dataset = load_dataset(
        args.dataset_name,
        None,
        cache_dir=args.cache_dir,
        task="image-classification",
        use_auth_token=False,
    )
    image_processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        cache_dir=args.cache_dir,
        revision="main",
        use_auth_token=False,
    )

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    data_type = torch.half if args.fp16 else torch.float32
    _test_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
            ConvertImageDtype(data_type)
        ]
    )

    def test_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_test_transforms(pil_img.convert("RGB")) for pil_img in
                                         example_batch["image"]]
        return example_batch

    dataset["test"].set_transform(test_transforms)

    dataloader = build_data_loader(
        dataset["test"],
        args.micro_batch_size,
        num_workers=args.num_workers,
        drop_last=(mpu.get_data_parallel_world_size() > 1),
        shuffle=False,
    )

    def metrics_func(model, epoch):
        print_rank_0("calculating metrics ...")
        if args.is_skip_compression_inference:
            args.skip_compression = True
        calculate_correct_answers(model, dataloader, epoch)
        if args.is_skip_compression_inference:
            args.skip_compression = False

    return metrics_func


def accuracy_func_provider():
    """Provide function that calculates accuracies."""
    args = get_args()
    data_path = args.data_path
    crop_size = (args.img_h, args.img_w)

    # Build dataloaders.
    val_data_path = data_path[1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_val = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = datasets.ImageFolder(root=val_data_path, transform=transform_val)

    dataloader = build_data_loader(
        dataset,
        args.micro_batch_size,
        num_workers=args.num_workers,
        drop_last=(mpu.get_data_parallel_world_size() > 1),
        shuffle=False
    )

    def metrics_func(model, epoch):
        print_rank_0("calculating metrics ...")
        correct, total = calculate_correct_answers(model, dataloader, epoch)
        percent = float(correct) * 100.0 / float(total)
        print_rank_last(
            " >> |epoch: {}| overall: correct / total = {} / {} = "
            "{:.4f} %".format(epoch, correct, total, percent)
        )

    return metrics_func


def calculate_correct_answers(model, dataloader, epoch):
    """Calculate correct over total answers"""

    args = get_args()
    forward_backward_func = get_forward_backward_func()
    for m in model:
        m.eval()

    saved_micro_batch_size = args.micro_batch_size
    saved_global_batch_size = args.global_batch_size

    ds = dataloader.dataset
    if hasattr(ds, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        sample_multiplier = ds.sample_multiplier
    else:
        sample_multiplier = 1

    micro_batch_size_times_data_parallel = args.orig_micro_batch_size * args.data_parallel_size
    num_micro_batches = args.orig_global_batch_size // micro_batch_size_times_data_parallel

    def loss_func(labels, output_tensor):
        logits = output_tensor

        loss_dict = {}
        # Compute the correct answers.
        predicted = torch.argmax(logits, dim=-1)
        corrects = (predicted == labels).float()
        # Add to the counters.
        loss_dict['total'] = labels.size(0)
        loss_dict['correct'] = corrects.sum().item()

        return 0, loss_dict

    #defined inside to capture output_predictions
    def correct_answers_forward_step(batch, model):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        images, labels = process_batch(batch_)

        # Forward model.
        output_tensor = model(images)

        return output_tensor, partial(loss_func, labels)

    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        for _, batch in enumerate(dataloader):

            actual_batch_size = len(batch['labels'])
            # ... applying sample_multiplier if necessary
            args.micro_batch_size = actual_batch_size * sample_multiplier
            args.global_batch_size = actual_batch_size * sample_multiplier * num_micro_batches

            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model,
                                               optimizer=None, timers=None, forward_only=True)

            for loss_dict in loss_dicts:
                total += loss_dict['total']
                correct += loss_dict['correct']

    for m in model:
        m.train()

    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.LongTensor([correct, total])
        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.
        correct_ans = unreduced[0].item()
        total_count = unreduced[1].item()
        percent = float(correct_ans) * 100.0 / float(total_count)
        print_rank_last(
            " >> |epoch: {}| overall: correct / total = {} / {} = "
            "{:.4f} %".format(epoch, correct_ans, total_count, percent)
        )

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
                filename = "../vit_results/" + str(args.dataset_name) \
                           + "_" + str(args.epochs) \
                           + "_" + str(args.lr) \
                           + "_" + str(args.lr_decay_style) \
                           + "_" + str(args.num_layers) \
                           + "_" + str(args.hidden_size) \
                           + "_" + str(args.num_attention_heads) \
                           + "_" + str(args.patch_size) \
                           + "_" + str(args.is_pipeline_compress) \
                           + "_" + str(args.pipeline_compress_method) \
                           + "_" + str(args.pipeline_qr_r) \
                           + "_" + str(args.is_tensor_compress) \
                           + "_" + str(args.tensor_compress_method) \
                           + "_" + str(args.tensor_qr_r) + ".txt"
                with open(filename, 'a') as f:
                    f.write(str(percent) + '\n')
