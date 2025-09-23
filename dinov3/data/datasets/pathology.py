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
            self._entry_dataset = pd.read_csv(os.path.join(self.root, "entry.csv"))
        if self._entry_slide is None:
            self._entry_slide = {}
            for _, row in self._entry_dataset.iterrows():
                dataset_name = row["dataset_name"]
                patches_path = row["patches_path"]
                self._entry_slide[dataset_name] = sorted(os.listdir(patches_path))
        
        assert self._entries is not None
        assert self._entry_dataset is not None
        assert self._entry_slide is not None
        return self._entries, self._entry_dataset, self._entry_slide

    def get_image_data(self, index: int) -> bytes:
        entries, entry_dataset, entry_slide = self._get_entries()
        actual_index = entries[index]["actual_index"]
        dir_index = entries[index]["dir_index"]
        dataset_name = entries[index]["dataset_name"]
        dataset_path = entry_dataset[entry_dataset["dataset_name"] == dataset_name]["patches_path"].values[0]
        patches_path = entry_slide[dataset_name][dir_index]


        image_full_path = os.path.join(dataset_path, patches_path,f"{patches_path}_patch_{actual_index}.jpeg")
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
        max_dataset_name_length = 0

        for _, row in entry_df.iterrows():
            dataset_name = row["dataset_name"]
            patches_dirs_path = row["patches_path"]
            max_dataset_name_length = max(len(dataset_name), max_dataset_name_length)
            patches_dirs = os.listdir(patches_dirs_path)
            for patches_dir in patches_dirs:
                patches_path = os.path.join(patches_dirs_path, patches_dir)
                sample_count += len(os.listdir(patches_path))

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("dir_index", "<u4"),
                ("dataset_name", f"U{max_dataset_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count+1, dtype=dtype)

        pbar = tqdm(total=sample_count, desc="Total progress")
        count_idx = 0
        for _, row in entry_df.iterrows():
            dataset_name = row["dataset_name"]
            patches_dirs_path = row["patches_path"]
            patches_dirs = os.listdir(patches_dirs_path)
            patches_dirs.sort()
            for dir_index, patches_dir in enumerate(patches_dirs):
                patches_path = os.path.join(patches_dirs_path, patches_dir)
                for patch_file in os.listdir(patches_path):
                    if count_idx+1 >= sample_count:
                        break
                    if not patch_file.endswith(".png"):
                        continue
                    actual_index = int(patch_file.split(".")[0].split("_")[-1])
                    entries_array[count_idx]["actual_index"] = actual_index
                    entries_array[count_idx]["dir_index"] = dir_index
                    entries_array[count_idx]["dataset_name"] = dataset_name
                    count_idx += 1
                    pbar.update(1)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def dump_extra(self) -> None:
        self._dump_entries()
