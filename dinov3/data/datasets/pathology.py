# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import csv
import logging
import os
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")
_Target = int


class _Split(Enum):
    TRAIN = "train"

class Pathology(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Pathology.Split",
        root: str,
        extra: str,
        sshid: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._extra_root = extra
        self.sshid = sshid,
        self._split = split

        self._entries = None
        self._entry_dataset = None
        self._entry_slide = None

    @property
    def split(self) -> "Pathology.Split":
        return self._split
    
    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"
    
    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)
    
    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        if self._entry_dataset is None:
            _entry_dataset_df = pd.read_csv(os.path.join(self.root, "entry.csv"))
            self._entry_dataset = np.array(_entry_dataset_df["patches_path"].tolist()).astype(np.bytes_)
        if self._entry_slide is None:
            _entry_slide = []
            max_slide_num = 0
            for idx, row in _entry_dataset_df.iterrows():
                patches_path = row["patches_path"]
                max_slide_num = max(max_slide_num, len(os.listdir(patches_path)))
            for idx, row in _entry_dataset_df.iterrows():
                patches_path = row["patches_path"]

                if self.sshid[0]=="11":
                    if patches_path.startswith("/data"):
                        patches_path = patches_path.replace("/data", "/nfs/data13", 1)
                    elif patches_path.startswith("/nfs/data11"):
                        patches_path = patches_path.replace("/nfs/data11", "/data", 1)

                pad_slide_num = max_slide_num - len(os.listdir(patches_path))
                _entry_slide.append(sorted(os.listdir(patches_path)) + [""]*pad_slide_num)
            self._entry_slide = np.array(_entry_slide).astype(np.bytes_)
        
        assert self._entries is not None
        assert self._entry_dataset is not None
        assert self._entry_slide is not None
        return self._entries, self._entry_dataset, self._entry_slide
    
    def get_image_data(self, index: int) -> bytes:
        entries, entry_dataset, entry_slide = self._get_entries()
        actual_index = entries[index]["actual_index"]
        dir_index = entries[index]["dir_index"]
        dataset_index = entries[index]["dataset_index"]
        dataset_path = str(entry_dataset[dataset_index],encoding="utf-8")
        slide_name = str(entry_slide[dataset_index][dir_index],encoding="utf-8")

        if self.sshid[0]=="11":
            if dataset_path.startswith("/data"):
                dataset_path = dataset_path.replace("/data", "/nfs/data13", 1)
            elif dataset_path.startswith("/nfs/data11"):
                dataset_path = dataset_path.replace("/nfs/data11", "/data", 1)

        image_full_path = os.path.join(dataset_path, slide_name,f"{slide_name}_patch_{actual_index}.jpeg")
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data
    
    def get_target(self, index: int) -> Optional[Target]:
        return None

    def __len__(self) -> int:
        entries, _, _ = self._get_entries()
        print(self.split,len(entries))
        #assert len(entries) == self.split.length
        return len(entries)
    

    def _dump_entries(self) -> None:
        
        entry_path = "entry.csv"
        logger.info(f'loading entries from "{entry_path}"')
        entry_df = pd.read_csv(os.path.join(self.root, entry_path))

        sample_count = 0

        for _, row in entry_df.iterrows():
            patches_dirs_path = row["patches_path"]
            patches_dirs = os.listdir(patches_dirs_path)
            for patches_dir in patches_dirs:
                patches_path = os.path.join(patches_dirs_path, patches_dir)
                sample_count += len(os.listdir(patches_path))

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("dir_index", "<u4"),
                ("dataset_index", f"<u4"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        pbar = tqdm(total=sample_count, desc="Total progress")
        count_idx = 0
        for idx, row in entry_df.iterrows():
            dataset_index = idx
            patches_dirs_path = row["patches_path"]
            patches_dirs = os.listdir(patches_dirs_path)
            patches_dirs.sort()
            for dir_index, patches_dir in enumerate(patches_dirs):
                patches_path = os.path.join(patches_dirs_path, patches_dir)
                for patch_file in os.listdir(patches_path):
                    if count_idx+1 > sample_count:
                        break
                    if not patch_file.endswith(".jpeg"):
                        continue
                    actual_index = int(patch_file.split(".")[-2].split("_")[-1])
                    entries_array[count_idx]["actual_index"] = actual_index
                    entries_array[count_idx]["dir_index"] = dir_index
                    entries_array[count_idx]["dataset_index"] = dataset_index
                    count_idx += 1
                    pbar.update(1)

        import math
        epoch_length = sample_count // (64 * 4 * 100) #96
        epoch_length = math.ceil(epoch_length / 50) * 50
        print(f'saving entries to "{self._entries_path}"')
        print("为了让模型至少见过每张图片一次，需满足：")
        print("batch_size * epochs * epoch_length > dataset_size")
        print("其中：batch_size = batch_size_per_gpu * gpu_num = 64 * 4 = 384; epochs = 100")
        print(f"因此：epoch_length 为 dataset_size / batch_size / epochs，约为 {epoch_length}，请填入配置文件的train.OFFICIAL_EPOCH_LENGTH")
        self._save_extra(entries_array, self._entries_path)

    def dump_extra(self) -> None:
        self._dump_entries()
