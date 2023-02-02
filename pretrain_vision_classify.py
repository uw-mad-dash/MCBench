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

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, mpu, print_rank_0
from megatron.data.vit_dataset import build_train_valid_datasets, build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.model.vision.classification import VitClassificationModel
from megatron.model.vision.classification import MitClassificationModel
from megatron.training import pretrain_vit
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


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()

    if args.vision_backbone_type == 'vit':
        print_rank_0("building VIT model ...")
        model = VitClassificationModel(num_classes=args.num_classes,
                                       pre_process=pre_process,
                                       post_process=post_process)
    elif args.vision_backbone_type == 'mit':
        print_rank_0("building MIT model ...")
        model = MitClassificationModel(num_classes=args.num_classes,
                                       pre_process=pre_process,
                                       post_process=post_process)
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))
    return model


def get_batch(data_iterator):
    """Build the batch."""
    args = get_args()
    # Items and their type.
    keys = ['images', 'labels']
    if args.fp16:
        datatype = torch.float16
    else:
        datatype = torch.float32

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        data_dict = {}
        data_dict['images'] = data['pixel_values'].to(datatype)
        data_dict['labels'] = data['labels'].to(datatype)
    else:
        data = None
        data_dict = None

    data_b = mpu.broadcast_data(keys, data_dict, datatype)
    images = data_b['images'].to(datatype)
    labels = data_b['labels'].to(torch.int64)

    return images, labels


def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
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
    # If we don't have a validation split, split off a percentage of train as validation.
    # args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    # if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
    #     split = dataset["train"].train_test_split(args.train_val_split)
    #     dataset["train"] = split["train"]
    #     dataset["validation"] = split["test"]
    # print("\033[31m before transform dataset[train][0]: \033[0m", dataset["train"][0])

    ### Here, we need to change the ImageProcessor for vit-base, vit-large and vit-huge
    if args.vision_backbone_type == 'vit':
        image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            cache_dir=args.cache_dir,
            revision="main",
            use_auth_token=False,
        )
    else:
        raise ValueError("vision_backbone_type is not implemented")

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
            # Resize(size),
            # CenterCrop(size),
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
        example_batch["pixel_values"] = [_test_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if "train" not in dataset:
        raise ValueError("--do_train requires a train dataset")
    else:
        dataset["train"].set_transform(train_transforms)

    if "valid" not in dataset:
        raise ValueError("--do_eval requires a validation dataset")
    else:
        dataset["valid"].set_transform(test_transforms)
    # print("\033[31m before select dataset[train][0]: \033[0m", dataset["train"][0])
    train_ds = dataset["train"]
    ### replace dataset["train"] with dataset["validation"]
    valid_ds = dataset["valid"]
    # print("\033[31m train_ds[0]: \033[0m", train_ds[0])
    # print("\033[31m train_ds[1]: \033[0m", train_ds[1])
    # print("\033[31m train_ds[2]: \033[0m", train_ds[2])
    # print("\033[31m train_ds[3]: \033[0m", train_ds[3])

    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":

    pretrain_vit(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )
