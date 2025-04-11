"""
Dataloader for seismic datasets.

"""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import Union, Tuple, List, Dict
import os
import shutil
import h5py
from numbers import Integral
from tqdm import tqdm
from pathlib import Path


class FixSizedDict(dict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: np.ndarray):
        dict.__setitem__(self, key, value)
        if self._maxlen > 0 and len(self) > self._maxlen:
            self.pop(next(iter(self)))


class UnconditionalSeismic3D(Dataset):
    """
    Implement: __len__, __getitem__, _move_to_scratch
    (normalize_input, denormalize_input, normalize_output, denormalize_output)
    """
    def __init__(
        self,
        store_dirpath: str,
        move_to_local_scratch: bool = True,
        samples_per_file: int = 100,
        channels: List[str] = ['uE'],  # [uE, uN, uZ]
        input_shape: Tuple[int, int, int] = (32, 32, 128),  # x, y, t
        max_files: int = None,
        cache_size: int = 10,
        normalize: bool = True,
        allow_stat_calculation: bool = False,
    ):
        # Locate dataset and move to scratch if requested
        assert os.path.exists(store_dirpath), f'Dataset directory {store_dirpath} does not exist'
        self.store_dirpath = store_dirpath
        if move_to_local_scratch:
            self.scratch_dirpath = self._move_to_local_scratch(store_dirpath)
            self.using_local_scratch = True
        else:
            self.scratch_dirpath = None
            self.using_local_scratch = False

        # Initialize dataset parameters
        self.channels = channels
        self.input_shape = input_shape
        self.data_fpaths = sorted(  # Sort by shard no
            list(Path(self._get_datapath()).glob('shard*.h5')), 
            key=lambda x: int(x.stem.split('shard')[1])
        )[:max_files]
        self.num_files = len(self.data_fpaths)
        self.samples_per_file = samples_per_file
        self.num_samples = self.num_files * self.samples_per_file
        self.normalize = normalize

        # Initialize cache
        self.cached_data = FixSizedDict(maxlen=cache_size)

        # Initialize parameters needed for GenCFD
        self.output_shape = input_shape
        self.input_channel = 1
        self.output_channel = 1
        self.spatial_resolution = input_shape

        # Load/calculate stats for normalization
        if self.normalize:
            if not os.path.exists(os.path.join(self._get_datapath(), 'norm_stats.h5')):
                if not allow_stat_calculation:
                    raise ValueError(f'Normalization statistics not found in {self._get_datapath()}.'
                    ' Turn on allow_stat_calculation to calculate them.')           
                print(f'Did not find normalization statistics in {self._get_datapath()}. Calculating them...')
                self._calculate_and_store_norm_stats()
            self._load_norm_stats()

    def _get_datapath(self):
        if self.using_local_scratch:
            return self.scratch_dirpath
        else:
            return self.store_dirpath

    def _load_norm_stats(self):
        """Load the normalization statistics from the dataset directory."""
        self.data_mean = torch.zeros(len(self.channels), *self.input_shape)
        self.data_std = torch.zeros(len(self.channels), *self.input_shape)

        with h5py.File(os.path.join(self._get_datapath(), 'norm_stats.h5'), 'r') as f:
            for i, channel in enumerate(self.channels):
                self.data_mean[i] = torch.from_numpy(f[f'{channel}_mean'][:])
                self.data_std[i] = torch.from_numpy(f[f'{channel}_std'][:])

    def _calculate_and_store_norm_stats(self):
        """Calculate the normalization statistics for the dataset."""

        print(f'Calculating normalization statistics for {self._get_datapath()}')

        all_channels = ['uE', 'uN', 'uZ']
        
        # Initialize counters for 1st and 2nd moment
        cnt = 0
        m1 = torch.zeros(len(all_channels), *self.input_shape)
        m2 = torch.zeros(len(all_channels), *self.input_shape)
        
        # Iterate through all files to calculate running statistics
        for fpath in tqdm(list(Path(self._get_datapath()).glob('shard*.h5')), 
                desc='Calculating normalization statistics'):
            trace_data = self._load_file(fpath, all_channels)
            cnt += trace_data.shape[0]
            m1 += trace_data.sum(dim=0)
            m2 += (trace_data ** 2).sum(dim=0)

        # Calculate final mean and standard deviation
        mean = m1 / cnt
        std = torch.sqrt(m2/(cnt-1) - (mean**2) / (cnt*(cnt-1)))

        # Store the normalization statistics (in store for permanent storage)
        print(f"Storing normalization statistics in {self.store_dirpath}")
        with h5py.File(os.path.join(self.store_dirpath, 'norm_stats.h5'), 'w') as f:
            for i, channel in enumerate(all_channels):
                f.create_dataset(f'{channel}_mean', data=mean[i])
                f.create_dataset(f'{channel}_std', data=std[i])

        # Copy the normalization statistics to the scratch directory
        if self.using_local_scratch:
            print(f"Copying normalization statistics to {self.scratch_dirpath}")
            shutil.copy(os.path.join(self.store_dirpath, 'norm_stats.h5'), os.path.join(self.scratch_dirpath, 'norm_stats.h5'))                    

    def _move_to_local_scratch(self, store_dirpath: str, scratch_superdir: str = "TMPDIR") -> str:
        """Copy the specified file to the local scratch directory if needed."""

        # Ensure scratch_dir is correctly resolved
        if scratch_superdir == "TMPDIR":    # Default to '/tmp' if TMPDIR is undefined
            scratch_superdir = os.environ.get("TMPDIR", "/tmp")

        # Construct the full scratch destination path
        scratch_dirpath = os.path.join(scratch_superdir, os.path.basename(store_dirpath))

        RANK = int(os.environ.get("LOCAL_RANK", -1))

        # Only copy if the file doesn't exist at the destination
        if not os.path.exists(scratch_dirpath) and (RANK == 0 or RANK == -1):
            print(f"Copying {store_dirpath} \nto {scratch_dirpath}...")
            shutil.copytree(store_dirpath, scratch_dirpath)
            print("Finished data copy.")

        if dist.is_initialized():
            dist.barrier(device_ids=[RANK])

        return scratch_dirpath
    
    def _idx_to_fpath_and_loc(self, index: int) -> Tuple[Path, int]:
        """Convert a sample index to a file name and location."""
        assert index < self.num_samples, f'Index {index} is out of bounds for the dataset'

        file_idx = index // self.samples_per_file
        loc_idx = index % self.samples_per_file

        return self.data_fpaths[file_idx], loc_idx
    
    def _load_file(self, fpath: Path, channels: List[str] = None) -> torch.Tensor:
        """
        Load a file from the dataset.
        
        Returns a tensor of shape (self.samples_per_file, len(self.channels), *self.input_shape)
        """
        channels = channels if channels is not None else self.channels

        trace_data = np.zeros((self.samples_per_file, len(self.channels), *self.input_shape))

        with h5py.File(fpath, 'r') as f:
            for i, trace_name in enumerate(self.channels):
                trace_dict = f[trace_name]
                for j in range(self.samples_per_file):
                    trace_data[j, i] = trace_dict[f'sample{j}'][:]
        return torch.tensor(trace_data, dtype=torch.float32)

    def __getitem__(self, index: Integral, verbose: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset. Holding single shard in memory and if needed load from disk.
        
        Called when indexing dataset in any way directly, e.g. `dataset[0]`, `dataset[0:10]`, `dataset[[0, 1, 2]]`. 
        Only Pytorch BatchSampler (and other classed/functions?) will automatically call `__getitems__` (sic!)
        for batching. Therefore calling `__getitems__` internally to catch cases like `dataset[0:10]` or `dataset[[0, 1, 2]]`.
        """

        # Handle multi-index calls
        if not isinstance(index, Integral):
            return self.__getitems__(index, verbose=verbose)

        # Load data from cache or file
        fpath, loc_idx = self._idx_to_fpath_and_loc(index)
        if verbose: print(f"Loading file {fpath} at location {loc_idx}")
        if fpath.stem not in self.cached_data:
            if verbose: print(f"Loading file {fpath} into cache")
            self.cached_data[fpath.stem] = self._load_file(fpath)
        elif verbose: 
            print(f"File {fpath} already in cache")
        trace_data = self.cached_data[fpath.stem][loc_idx]  # Result has shape <self.input_shape>

        if self.normalize:
            trace_data = (trace_data - self.data_mean) / self.data_std

        return {
            "target_cond": trace_data,
        }
    
    def __getitems__(self, indices: Union[int, slice, list], verbose: bool = False) -> List[Dict[str, torch.Tensor]]:
        """ 
        Used for batching. TODO: Implement batching.
        """

        assert not isinstance(indices, int), \
            "Single integer indices are not supported. Use `__getitem__` instead."

        if isinstance(indices, slice):
            indices = range(indices.start or 0, indices.stop or len(self), indices.step or 1)

        return [self.__getitem__(i, verbose=verbose) for i in indices]
    
    def __len__(self) -> int:
        return self.num_samples
