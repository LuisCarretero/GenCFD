"""
Dataloader for seismic datasets.

"""

import numpy as np
import torch
import torch.distributed as dist
from typing import Union, Tuple, Any, List, Dict
import os
import shutil
import h5py


class FixSizedDict(dict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: np.ndarray):
        dict.__setitem__(self, key, value)
        if self._maxlen > 0 and len(self) > self._maxlen:
            self.pop(next(iter(self)))


class UnconditionalSeismic3D:
    """
    Implement: __len__, __getitem__, _move_to_scratch
    (normalize_input, denormalize_input, normalize_output, denormalize_output)
    """
    def __init__(
        self,
        dataset_dirpath: str,
        move_to_local_scratch: bool = True,
        samples_per_file: int = 100,
        trace_name: str = 'vE',
        input_shape: Tuple[int, int, int] = (32, 32, 128),  # x, y, t
    ):
        # Locate dataset and move to scratch if requested
        assert os.path.exists(dataset_dirpath), f"Dataset directory {dataset_dirpath} does not exist"
        if move_to_local_scratch:
            self.dataset_dirpath = self._move_to_local_scratch(dataset_dirpath)
        else:
            self.dataset_dirpath = dataset_dirpath

        # Initialize dataset parameters
        self.trace_name = trace_name
        self.input_shape = input_shape
        self.file_list = os.listdir(self.dataset_dirpath)
        self.num_files = len(self.file_list)
        self.samples_per_file = samples_per_file
        self.num_samples = self.num_files * self.samples_per_file

        # Initialize cache
        self.cached_data = FixSizedDict(maxlen=10)

    
    def _move_to_local_scratch(self, dataset_dirpath: str, scratch_dir: str = "TMPDIR") -> str:
        """Copy the specified file to the local scratch directory if needed."""

        # Ensure scratch_dir is correctly resolved
        if scratch_dir == "TMPDIR":    # Default to '/tmp' if TMPDIR is undefined
            scratch_dir = os.environ.get("TMPDIR", "/tmp")

        # Construct the full destination path
        dest_path = os.path.join(scratch_dir, os.path.basename(dataset_dirpath))

        RANK = int(os.environ.get("LOCAL_RANK", -1))

        # Only copy if the file doesn't exist at the destination
        if not os.path.exists(dest_path) and (RANK == 0 or RANK == -1):
            print(f"Start copying {dataset_dirpath} to {dest_path}...")
            shutil.copy(dataset_dirpath, dest_path)
            print("Finished data copy.")

        if dist.is_initialized():
            dist.barrier(device_ids=[RANK])

        return dest_path
    
    def _idx_to_fname_and_loc(self, index: int) -> Tuple[str, int]:
        """Convert a sample index to a file name and location."""
        assert index < self.num_samples, f"Index {index} is out of bounds for the dataset"

        file_idx = index // self.samples_per_file
        loc_idx = index % self.samples_per_file

        return self.file_list[file_idx], loc_idx
    
    def _load_file(self, fname: str) -> np.ndarray:
        """Load a file from the dataset."""

        with h5py.File(os.path.join(self.dataset_dirpath, fname), 'r') as f:
            trace_dict = f[self.trace_name]
            trace_data = np.zeros((self.samples_per_file, *self.input_shape))
            for i in range(self.samples_per_file):
                trace_data[i] = trace_dict[f'sample{i}'][:]
        return trace_data

    
    def __getitem__(self, index, verbose: bool = False) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset. Holding single shard in memory and if needed load from disk."""

        fname, loc_idx = self._idx_to_fname_and_loc(index)

        if verbose: print(f"[INFO] Getting item {index} from {fname} at location {loc_idx}")
        if fname not in self.cached_data:
            if verbose: print(f"[INFO] File not cached, loading from disk...")
            self.cached_data[fname] = self._load_file(fname)
        else:
            if verbose: print(f"[INFO] File cached, loading from memory...")

        trace_data = self.cached_data[fname][loc_idx]  # Result has shape <self.input_shape>

        return {
            "lead_time": None,
            "initial_cond": None,
            "target_cond": torch.tensor(trace_data, dtype=torch.float32),
        }
    
    def __len__(self) -> int:
        return self.num_samples
