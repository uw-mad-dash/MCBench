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

"""Vision-classification finetuning/evaluation."""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers
from megatron import print_rank_0
from megatron.model.vision.classification import VitClassificationModel
from megatron.data.vit_dataset import build_train_valid_datasets
from tasks.vision.classification.eval_utils import accuracy_func_provider_vit
from tasks.vision.finetune_utils import finetune
from megatron.utils import average_losses_across_data_parallel_group

from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ConvertImageDtype,
)


def classification():
    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()

        print_rank_0(
            "> building train, validation, and test datasets " "for VIT ..."
        )
        dataset = load_dataset(
            args.dataset_name,
            None,
            cache_dir=args.cache_dir,
            task="image-classification",
            use_auth_token=True,
        )

        ### Here, we need to change the ImageProcessor for vit-base, vit-large and vit-huge
        if args.pretrained_checkpoint.split("/")[1].endswith("base_patch16"):
            pretrain_name = "google/vit-base-patch16-224-in21k"
        elif args.pretrained_checkpoint.split("/")[1].endswith("base_patch32"):
            pretrain_name = "google/vit-base-patch32-224-in21k"
        elif args.pretrained_checkpoint.split("/")[1].endswith("large_patch16"):
            pretrain_name = "google/vit-large-patch16-224-in21k"
        elif args.pretrained_checkpoint.split("/")[1].endswith("large_patch32"):
            pretrain_name = "google/vit-large-patch32-224-in21k"
        elif args.pretrained_checkpoint.split("/")[1].endswith("huge_patch14"):
            pretrain_name = "google/vit-huge-patch14-224-in21k"
        else:
            raise ValueError("pretrain_name is wrong!")

        image_processor = ViTImageProcessor.from_pretrained(
            pretrain_name,
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
        _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
                ConvertImageDtype(data_type)
            ]
        )
        _test_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
                ConvertImageDtype(data_type)
            ]
        )

        def train_transforms(example_batch):
            """Apply _train_transforms across a batch."""
            example_batch["pixel_values"] = [
                _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]
            return example_batch

        def test_transforms(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [_test_transforms(pil_img.convert("RGB")) for pil_img in
                                             example_batch["image"]]
            return example_batch

        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        else:
            dataset["train"].set_transform(train_transforms)

        if "test" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        else:
            dataset["test"].set_transform(test_transforms)

        train_ds = dataset["train"]
        valid_ds = dataset["test"]

        print_rank_0("> finished creating VIT datasets ...")

        return train_ds, valid_ds

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0("building classification model for ImageNet ...")

        return VitClassificationModel(num_classes=args.num_classes, finetune=True,
                                      pre_process=pre_process, post_process=post_process)

    def process_batch(batch):
        """Process batch and produce inputs for the model."""
        images = batch['pixel_values'].cuda().contiguous()
        labels = batch['labels'].cuda().contiguous()
        return images, labels

    def cross_entropy_loss_func(labels, output_tensor):
        logits = output_tensor

        # Cross-entropy loss.
        loss = F.cross_entropy(logits.contiguous().float(), labels)

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}

    def _cross_entropy_forward_step(batch, model):
        """Simple forward step with cross-entropy loss."""
        timers = get_timers()

        # Get the batch.
        timers("batch generator").start()
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        images, labels = process_batch(batch_)
        timers("batch generator").stop()

        # Forward model.
        output_tensor = model(images)
      
        return output_tensor, partial(cross_entropy_loss_func, labels)

    """Finetune/evaluate."""
    finetune(
        train_valid_datasets_provider,
        model_provider,
        forward_step=_cross_entropy_forward_step,
        end_of_epoch_callback_provider=accuracy_func_provider_vit,
    )

def main():
    classification()

