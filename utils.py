#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pprint import pformat
from typing import Dict, List, Optional, Tuple, Union, Callable, TypeVar, Any
from einops import rearrange
from dataclasses import dataclass, field, asdict

from ignite.metrics import Loss, Precision, Recall, RunningAverage, Accuracy, EpochMetric
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torchaudio import transforms as audio_transforms
import torch_audiomentations as wavtransforms

R = TypeVar('R')

@dataclass
class MAEConfig:
    train_data: str
    cv_data : str
    outputpath: str = 'experiments/mae/'
    n_saved: int = 4
    epoch_length: Optional[int] = None
    epochs: int = 40
    batch_size: int = 32
    eval_batch_size: int = batch_size
    num_workers: int = 4
    chunk_length: float = 10.0 # in seconds
    average: bool = True # Average topk
    model: str = 'mae_audiotransformer_tiny'
    model_args: Dict[str, Any] = field(default_factory=dict)

    valid_every: int = 1 #When to run validation


    mask_ratio: float = 0.75
    optimizer: str = 'Adam8bit'
    optimizer_args: Dict[str, Any] = field(default_factory=lambda: {
        'lr': 0.0002,
        'weight_decay': 0.0000005
    })
    decay_steps: Optional[
        int] = None  # Decay over the entire length of training
    decay_frac: Optional[float] = 0.01
    warmup_iters: Optional[int] = None
    warmup_epochs: Optional[int] = 3

    use_scheduler: bool = True
    shuffle: bool = True
    logfile:str = 'train.log'
    pretrained_path: Optional[str] = None
    basename: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class AudiosetConfig:
    train_data: str
    cv_data: str
    psl_data: str # Parquet formatted data, can be downloaded
    outputpath: str = 'experiments/audioset'

    n_saved: int = 4
    valid_every: int = 1
    epoch_length: Optional[int] = None
    epochs: int = 150
    warmup_iters: Optional[int] = None
    warmup_epochs: int = 5
    #  Sampler Kwargs
    sampler: Optional[str] = 'balanced'
    replacement: bool = True
    num_samples: int = 200_000

    batch_size: int = 32
    eval_batch_size: int = batch_size
    num_workers: int = 4
    mixup_alpha: Optional[float] = None
    use_scheduler: bool = True
    num_classes: int = 527
    model: str = 'SAT_T_2s'
    model_args: Dict[str, Any] = field(default_factory=dict)
    pretrained_path: Optional[str] = None

    # Optimizer
    optimizer: str = 'Adam8bit'
    optimizer_args: Dict[str, Any] = field(default_factory=lambda: {
        'lr': 0.0005,
        'weight_decay': 0.00000005
    })
    decay_steps: Optional[int] = None
    decay_frac: Optional[float] = 0.1

    spectransforms: List = field(default_factory=list)
    wavtransforms: List = field(default_factory=list)
    chunk_length: Optional[float] = None
    basename: bool = True
    logfile: str = 'train.log'

    average: bool = True

    def to_dict(self):
        return asdict(self)


ALL_EVAL_METRICS = {
    'Accuracy':
    lambda: Accuracy(),
    'PositiveMultiClass_Accuracy':
    lambda: EpochMetric(compute_fn=compute_accuracy_with_noise),
    'Micro_Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=1),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Micro_Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average=None, zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_Precision':
    lambda: EpochMetric(lambda y_pred, y_tar: precision_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Recall':
    lambda: EpochMetric(lambda y_pred, y_tar: recall_score(
        y_tar.numpy(), y_pred.numpy(), average=None, zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Macro_F1':
    lambda: EpochMetric(lambda y_pred, y_tar: f1_score(
        y_tar.numpy(), y_pred.numpy(), average='macro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'Micro_F1':
    lambda: EpochMetric(lambda y_pred, y_tar: f1_score(
        y_tar.numpy(), y_pred.numpy(), average='micro', zero_division=0),
                        output_transform=lambda x:
                        ((x[0] > 0.2).float(), x[1]),
                        check_compute_fn=False),
    'AUC':
    lambda: EpochMetric(compute_roc_auc, check_compute_fn=False),
    'BCELoss':
    lambda: Loss(torch.nn.BCELoss()),
    'CELoss':
    lambda: Loss(torch.nn.CrossEntropyLoss()),
    'mAP':
    lambda: EpochMetric(lambda y_pred, y_tar: np.nanmean(
        average_precision_score(
            y_tar.to('cpu').numpy(), y_pred.to('cpu').numpy(), average=None)),
                        check_compute_fn=False),
    'mAP_transform':
    lambda output_transform:
    EpochMetric(output_transform=output_transform,
                compute_fn=lambda y_pred, y_tar: np.nanmean(
                    average_precision_score(y_tar.to('cpu').numpy(),
                                            y_pred.to('cpu').numpy(),
                                            average=None)),
                check_compute_fn=False),
    'AP':
    lambda: EpochMetric(lambda y_pred, y_tar: average_precision_score(
        y_tar.to('cpu').numpy(), y_pred.to('cpu').numpy(), average=None),
                        check_compute_fn=False),
    'lwlwrap':
    lambda: EpochMetric(calculate_overall_lwlrap_sklearn,
                        check_compute_fn=False),
    # metrics.Lwlwrap(),
    'ErrorRate':
    lambda: EpochMetric(lambda y_pred, y_tar: 1. - np.nan_to_num(
        accuracy_score(y_tar.to('cpu').numpy(),
                       y_pred.to('cpu').numpy())),
                        check_compute_fn=False),
}


def metrics(metric_names: List[str]) -> Dict[str, EpochMetric]:
    '''
    Returns metrics given some metric names
    '''
    return {met: ALL_EVAL_METRICS[met]() for met in metric_names}


class DictWrapper(object):

    def __init__(self, adict):
        self.dict = adict

    def state_dict(self):
        return self.dict

    def load_state_dict(self, state):
        self.dict = state


def load_pretrained(model: torch.nn.Module, trained_model: dict):
    if 'model' in trained_model:
        trained_model = trained_model['model']
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model[k].shape)
    }
    assert len(pretrained_dict) > 0, "Couldnt load pretrained model"
    # Found time positional embeddings ....
    if 'time_pos_embed' in trained_model.keys():
        pretrained_dict['time_pos_embed'] = trained_model['time_pos_embed']
        pretrained_dict['freq_pos_embed'] = trained_model['freq_pos_embed']

    logger.info(
        f"Loading {len(pretrained_dict)} Parameters for model {model.__class__.__name__}"
    )
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model


def parse_config_or_kwargs(config_file, default_args: Callable[..., R],
                           **kwargs) -> R:
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    return default_args(**arguments)


def parse_wavtransforms(transforms_dict: Dict) -> Callable:
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans_name, v in transforms_dict.items():
        transforms.append(getattr(wavtransforms, trans_name)(**v))

    return torch.nn.Sequential(*transforms)


def parse_spectransforms(transforms: Union[List, Dict]) -> Callable:
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    if isinstance(transforms, dict):
        return torch.nn.Sequential(*[
            getattr(audio_transforms, trans_name)(**v)
            for trans_name, v in transforms.items()
        ])
    elif isinstance(transforms, list):
        return torch.nn.Sequential(*[
            getattr(audio_transforms, trans_name)(**v) for item in transforms
            for trans_name, v in item.items()
        ])
    else:
        raise ValueError("Transform unknown")


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def mixup(x: torch.Tensor, lamb: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """

    x1 = rearrange(x.flip(0), 'b ... -> ... b')
    x2 = rearrange(x, 'b ... -> ... b')
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')


def read_tsv_data(datafile: str,
                  nrows: Optional[int] = None,
                  basename=True) -> pd.DataFrame:
    df = pd.read_csv(datafile, sep="\t", nrows=nrows)
    assert 'hdf5path' in df.columns and 'filename' in df.columns and 'labels' in df.columns
    if any(df['labels'].str.contains(';')):
        df['labels'] = df['labels'].str.split(';').map(
            lambda x: np.array(x, dtype=int))
    if basename:
        df['filename'] = df['filename'].str.rsplit('/').str[-1]
    return df


def read_psl_data(datafile: str, ) -> pd.DataFrame:
    psl_df = pd.read_parquet(datafile)
    psl_df['prob'] = psl_df['prob'].str.split(';').map(
        lambda x: np.array(x, dtype=np.float32)).reset_index(drop=True)
    psl_df['idxs'] = psl_df['idxs'].str.split(';').map(
        lambda x: np.array(x, dtype=np.int64)).reset_index(drop=True)
    return psl_df

def average_models(models: List[str]):
    model_res_state_dict = {}
    state_dict = {}
    has_new_structure = False
    for m in models:
        cur_state = torch.load(m, map_location='cpu')
        if 'model' in cur_state:
            has_new_structure = True
            model_params = cur_state.pop('model')
            # Append non "model" items, encoder, optimizer etc ...
            for k in cur_state:
                state_dict[k] = cur_state[k]
            # Accumulate statistics
            for k in model_params:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += model_params[k]
                else:
                    model_res_state_dict[k] = model_params[k]
        else:
            for k in cur_state:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += cur_state[k]
                else:
                    model_res_state_dict[k] = cur_state[k]

    # Average
    for k in model_res_state_dict:
        # If there are any parameters
        if model_res_state_dict[k].ndim > 0:
            model_res_state_dict[k] /= float(len(models))
    if has_new_structure:
        state_dict['model'] = model_res_state_dict
    else:
        state_dict = model_res_state_dict
    return state_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help="Output model (pytorch)")
    args = parser.parse_args()
    mdls = average_models(args.models)
    torch.save(mdls, args.output)
