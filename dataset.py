import torch
import numpy as np
from loguru import logger
from typing import Optional, Tuple, List,Any
import random
from pathlib import Path
from typing import Sequence
import pandas as pd
from typing import Dict, Sequence
from torch.multiprocessing import Queue
from h5py import File


INT_MAX = np.iinfo(np.int16).max

class WeakHDF5Dataset(torch.utils.data.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(
        self,
        data_frame: pd.DataFrame,
        num_classes: int,
    ):
        super(WeakHDF5Dataset, self).__init__()
        self._dataframe = data_frame
        self._datasetcache = {}
        self._len = len(self._dataframe)
        self._num_classes = num_classes

    def __len__(self) -> int:
        return self._len

    def __del__(self):
        for k, cache in self._datasetcache.items():
            cache.close()

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][:]
        if np.issubdtype(data.dtype, np.integer):
            data = (data/32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, hdf5path = self._dataframe.iloc[index][[
            'filename', 'labels', 'hdf5path'
        ]]
        # Init with all-Zeros classes i.e., nothing present
        target = torch.zeros(self._num_classes, dtype=torch.float32).scatter_(
            0, torch.as_tensor(label_idxs), 1)
        data = self._readdata(hdf5path, fname)
        return data, target, fname


class WeakRandomCropHDF5Dataset(WeakHDF5Dataset):

    def __init__(self,
                 data_frame,
                 chunk_length: float,
                 num_classes: int,
                 sample_rate: int = 16000,
                 smooth: bool = False):
        super(WeakRandomCropHDF5Dataset,
              self).__init__(data_frame, num_classes=num_classes)
        self._sr = sample_rate
        self.target_transform = self._target_transform_multilabel
        self._num_classes = num_classes
        self.chunk_length = int(chunk_length * sample_rate)
        self.smooth = smooth

    def _target_transform_multilabel(self, label_idx: List) -> torch.Tensor:
        target = torch.zeros(self._num_classes, dtype=torch.float32)

        target = target.scatter_(0, torch.as_tensor(label_idx), 1)
        return target

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data_shape = self._datasetcache[hdf5path][f"{fname}"].shape[-1]
        if data_shape > self.chunk_length:
            start_idx = random.randint(0, data_shape - self.chunk_length - 1)
            data = self._datasetcache[hdf5path][f"{fname}"][
                start_idx:start_idx + self.chunk_length]
        else:
            load_data = self._datasetcache[hdf5path][f"{fname}"][:]
            data = np.zeros(self.chunk_length, dtype=load_data.dtype)
            data_length = load_data.shape[-1]
            start_idx = 0
            #Randomly insert into array if longer
            if self.chunk_length - data_length > 0:
                start_idx = random.randint(0,
                                           self.chunk_length - data_length - 1)
            data[start_idx:start_idx + data_length] = load_data
        if np.issubdtype(data.dtype, np.integer):
            data = (data/32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, hdf5path = self._dataframe.iloc[index][[
            'filename', 'labels','hdf5path'
        ]]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        target= self.target_transform(label_idxs)
        data = self._readdata(hdf5path, fname)
        return data, target, fname

class WeakChunkedHDF5Dataset(WeakHDF5Dataset):
    def __init__(self, data_frame, num_classes:int, sample_rate:int =16000):
        super(WeakChunkedHDF5Dataset, self).__init__(data_frame, num_classes)
        self._sr = sample_rate
        self.target_transform = self._target_transform_multilabel
        self._num_classes = num_classes

    def _target_transform_singlelabel(self, label_idx:int) -> torch.Tensor:
        return torch.as_tensor(label_idx)

    def _target_transform_multilabel(self, label_idx:List) -> torch.Tensor:
        target = torch.zeros(self._num_classes,
                             dtype=torch.float32)
        if -1 in label_idx: label_idx.remove(-1)
        if len(label_idx) > 0:
            target = target.scatter_(
                0, torch.as_tensor(label_idx), 1)
        return target

    def _readdata(self, hdf5path: str, fname: str, from_time:int, to_time:int) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][from_time:to_time]
        if np.issubdtype(data.dtype, np.integer):
            data = (data/32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, from_time, to_time,  hdf5path = self._dataframe.iloc[index][[
            'filename', 'labels','from', 'to', 'hdf5path'
        ]]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        target= self.target_transform(label_idxs)

        from_time = int(from_time * self._sr)
        to_time = int(to_time * self._sr)
        data = self._readdata(hdf5path, fname, from_time, to_time)
        return data, target, fname


class UnlabeledRandomChunkedHDF5Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_df,
        chunk_length: float = 2.0,
        sample_rate: int = 16000,
        num_classes=527,
    ):
        super(UnlabeledRandomChunkedHDF5Dataset, self).__init__()
        self._dataframe = data_df
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.chunk_length_in_samples = int(self.sample_rate * self.chunk_length)
        self.num_classes = num_classes
        self._datasetcache = {}

        logger.info(
            f"Using random chunks of length {self.chunk_length_in_samples}")

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data_shape = self._datasetcache[hdf5path][f"{fname}"].shape[-1]
        if data_shape > self.chunk_length_in_samples:
            start_idx = random.randint(
                0, data_shape - self.chunk_length_in_samples - 1)
            data = self._datasetcache[hdf5path][f"{fname}"][
                start_idx:start_idx + self.chunk_length_in_samples]
        else:
            load_data = self._datasetcache[hdf5path][f"{fname}"][:]
            data = np.zeros(self.chunk_length_in_samples,
                            dtype=load_data.dtype)
            data[:load_data.shape[0]] = load_data
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, hdf5path = self._dataframe.iloc[index][['filename', 'hdf5path']]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        data = self._readdata(hdf5path, fname)
        return data, torch.zeros(self.num_classes), fname

    def __len__(self):
        return len(self._dataframe)

class UnlabeledHDF5Dataset(UnlabeledRandomChunkedHDF5Dataset):
    def __init__(self,*args,**kwrgs):
        super().__init__(*args, **kwrgs)

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data_shape = self._datasetcache[hdf5path][f"{fname}"].shape[-1]
        if data_shape > self.chunk_length_in_samples:
            start_idx = 0
            data = self._datasetcache[hdf5path][f"{fname}"][
                start_idx:start_idx + self.chunk_length_in_samples]
        else:
            load_data = self._datasetcache[hdf5path][f"{fname}"][:]
            data = np.zeros(self.chunk_length_in_samples,
                            dtype=load_data.dtype)
            data[:load_data.shape[0]] = load_data
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

class WeakHDF5DatasetTextLogits(torch.utils.data.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(
        self,
        data_frame: pd.DataFrame,
        num_classes: int = 527,
        sr: int = 16000,
    ):
        super().__init__()
        self._datasetcache = {}
        self._dataframe = data_frame
        self._sr = sr
        self._num_classes = num_classes

    def _readdata(self, hdf5path: str, fname: str, start: int,
                  end: int) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')

        start_idx = int(start * self._sr)
        end_idx = int(end * self._sr)
        data = self._datasetcache[hdf5path][f"{fname}"][start_idx:end_idx]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        fname, probs, idxs, start, end, hdf5path = self._dataframe.iloc[index][
            ['filename', 'prob', 'idxs', 'start', 'end', 'hdf5path']]

        soft_target = torch.zeros(self._num_classes,
                                  dtype=torch.float32).scatter_(
                                      0, torch.as_tensor(idxs),
                                      torch.as_tensor(probs,
                                                      dtype=torch.float32))
        target = torch.zeros(self._num_classes, dtype=torch.float32)
        data = self._readdata(hdf5path, fname, start, end)
        return data, target, soft_target, fname

    def __len__(self):
        return len(self._dataframe)


def pad(tensorlist: Sequence[torch.Tensor], padding_value: float = 0.):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim, ) + trailing_dims + (num_raw_samples, )
    out_tensor = torch.full(out_dims,
                            fill_value=padding_value,
                            dtype=torch.float32)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor, lengths


def sequential_pad(batches):
    datas, *targets, fnames = zip(*batches)
    targets = tuple(map(lambda x: torch.stack(x), targets))
    datas, lengths = pad(datas)
    return datas, *targets, torch.as_tensor(lengths), fnames


class BalancedSampler(torch.utils.data.WeightedRandomSampler):

    def __init__(self,
                 labels_df: pd.Series,
                 replacement:bool = True,
                 num_samples: Optional[int] = None,
                 offset: int = 100,
                 random_state=None):
        #labels_df is ideally a dataframe with some keys and labels as:
        #[0,11]
        #[5,100,200]
        self._random_state = np.random.RandomState(seed=random_state)
        single_labels_df = labels_df.copy().explode().reset_index()
        single_labels_df.columns = ['index', 'label']
        occurances = single_labels_df.groupby('label')['index'].apply(len)
        occurances = occurances.sort_index()
        weights = 1000. / (occurances + offset)  # Offset classes with low prob
        weights = weights.to_dict()
        sample_weights = labels_df.apply(
            lambda x: sum([weights[class_id] for class_id in x])).values
        num_samples = len(sample_weights) if num_samples is None else num_samples
        super().__init__(sample_weights, num_samples=num_samples, replacement=replacement)



if __name__ == "__main__":
    from tqdm import tqdm
    import utils
