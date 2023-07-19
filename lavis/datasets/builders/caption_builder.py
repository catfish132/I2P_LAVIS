"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from lavis.common.registry import registry
from lavis.datasets.datasets.video_caption_datasets import (
    VideoCaptionDataset,
    VideoCaptionEvalDataset,
)
from lavis.datasets.datasets.i2p_datasets import (
I2pDataset,
I2pEvalDataset
)

from lavis.datasets.datasets.room_datasets import (
RoomDataset,
RoomEvalDataset
)

from lavis.datasets.datasets.minicoco_datasets import (
MinicocoDataset,
MinicocoEvalDataset
)


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("nocaps")
class COCOCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls = NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }


@registry.register_builder("msrvtt_caption")
class MSRVTTCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_cap.yaml",
    }


@registry.register_builder("msvd_caption")
class MSVDCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_cap.yaml",
    }


@registry.register_builder("vatex_caption")
class VATEXCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoCaptionDataset
    eval_dataset_cls = VideoCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vatex/defaults_cap.yaml",
    }


@registry.register_builder("i2p")
class I2pBuilder(BaseDatasetBuilder):
    train_dataset_cls = I2pDataset
    eval_dataset_cls = I2pEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/i2p/default_i2p.yaml"
    }

@registry.register_builder("room")
class RoomBuilder(BaseDatasetBuilder):
    train_dataset_cls = RoomDataset
    eval_dataset_cls = RoomEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/room/default_room.yaml",
        "huaban": "configs/datasets/room/huaban_room.yaml"
    }

@registry.register_builder("minicoco")
class MinicocoBuilder(BaseDatasetBuilder):
    train_dataset_cls = MinicocoDataset
    eval_dataset_cls = MinicocoEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/minicoco/default_minicoco.yaml",
        "enhanced": "configs/datasets/minicoco/minicoco_enhanced.yaml"
    }