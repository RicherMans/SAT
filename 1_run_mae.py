from typing import List, Tuple, Dict, Any, Union, Callable, Iterable, Optional, TypeVar
from loguru import logger
from fire import Fire
import pandas as pd
import uuid
import numpy as np

import models
import dataset
import utils
import torch
import sys
import datetime
from pathlib import Path
import ignite
from ignite.contrib.handlers import ProgressBar, create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import (Checkpoint, DiskSaver, global_step_from_engine)


class MAELoss(torch.nn.Module):

    def __init__(self, norm_pix_loss=True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


logger.configure(handlers=[{
    "sink": sys.stdout,
    "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
    'level': 'DEBUG',
}])

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')


def transfer_to_device(batch: Iterable, device=DEVICE):
    return (x.to(device, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


def log_basic_info(outputdir, config_parameters):
    import os
    if 'HOSTNAME' in os.environ:
        logger.info(f"Running on host {os.environ['HOSTNAME']}")

    logger.info(f"Running on device {DEVICE}")
    logger.info(f"Storing output in {outputdir}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.current_device()}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
    for k, v in config_parameters.to_dict().items():
        logger.info(f"{k} : {v}")


def create_engine(engine_function: Callable,
                  evaluation_metrics: Optional[List[str]] = None):
    engine = Engine(engine_function)
    ProgressBar().attach(engine, output_transform=lambda x: x)

    if evaluation_metrics:
        eval_mets = utils.metrics(evaluation_metrics)
        for name, metric in eval_mets.items():
            metric.attach(engine, name)
    return engine


class RunnerMAE(object):

    def __init__(self, seed: int = 42, nthreads: int = 1):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(nthreads)

    def __setup(self,
                config: Union[Path, str],
                default_args=utils.MAEConfig,
                **override_kwargs) -> Tuple[Path, utils.MAEConfig]:
        config_parameters = utils.parse_config_or_kwargs(
            config, default_args=default_args, **override_kwargs)
        outputdir = Path(config_parameters.outputpath) / Path(
            config).stem / f"{config_parameters.model}" / "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
                uuid.uuid1().hex)
        outputdir.mkdir(exist_ok=True, parents=True)
        log_fname = config_parameters.logfile
        output_log = outputdir / log_fname
        logger.add(
            output_log,
            enqueue=True,
            level='INFO',
            format=
            "[<red>{level}</red> <green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
        )
        log_basic_info(outputdir, config_parameters)
        return outputdir, config_parameters

    def train(self, config: Union[str, Path], **overwrite_kwargs: Dict[str,
                                                                       Any]):
        outputdir, config_parameters = self.__setup(config, **overwrite_kwargs)
        epochs: int = config_parameters.epochs
        mask_ratio = config_parameters.mask_ratio

        model = getattr(
            models, config_parameters.model)(**config_parameters.model_args)
        logger.info(model)
        model = model.to(DEVICE).train()

        if config_parameters.optimizer == 'Adam8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                **config_parameters.optimizer_args)  # add bnb optimizer
        else:
            optimizer = getattr(torch.optim, config_parameters.optimizer)(
                model.parameters(), **config_parameters.optimizer_args)
        train_df = utils.read_tsv_data(config_parameters.train_data,
                                       basename=config_parameters.basename)
        cv_df = utils.read_tsv_data(config_parameters.cv_data,
                                    basename=config_parameters.basename)
        logger.info(
            f"Got {len(train_df)} train samples and {len(cv_df)} validation ones."
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset.UnlabeledRandomChunkedHDF5Dataset(
                data_df=train_df,
                chunk_length=config_parameters.chunk_length,
            ),
            batch_size=config_parameters.batch_size,
            num_workers=config_parameters.num_workers,
            shuffle=config_parameters.shuffle,
            collate_fn=dataset.sequential_pad)
        test_dataloader = torch.utils.data.DataLoader(
            dataset.UnlabeledHDF5Dataset(
                cv_df, chunk_length=config_parameters.chunk_length),
            batch_size=config_parameters.eval_batch_size,
            num_workers=config_parameters.num_workers,
            shuffle=False,
        )
        criterion = MAELoss().to(DEVICE)

        def _train(engine, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad(set_to_none=True)
                x, *_ = transfer_to_device(batch)
                pred, tar, mask = model(x, mask_ratio=mask_ratio)
                loss = criterion(pred, tar, mask)
                loss.backward()
                optimizer.step()
                return {
                    'total_loss': loss.item(),
                }

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, *_ = transfer_to_device(batch)
                pred, tar, mask = model(data, mask_ratio=mask_ratio)
                return {
                    'y_pred': pred,
                    'y': tar,
                    'criterion_kwargs': {
                        'mask': mask
                    }
                }

        def log_metrics(engine, title=None):
            results = engine.state.metrics
            output_str_list = [
                f"{title:<10} Results - Epoch : {train_engine.state.epoch:<4}"
            ] + [f"{metric} {results[metric]:<5.4f}" for metric in results
                 ] + [f"LR: {optimizer.param_groups[0]['lr']:.2e}"]
            logger.info(" ".join(output_str_list))

        train_engine = create_engine(_train)
        inference_engine = create_engine(_inference)

        ignite.metrics.Loss(criterion).attach(inference_engine, 'Loss')

        score_function = Checkpoint.get_default_score_fn(*['Loss', -1.0])
        checkpoint_saver = Checkpoint(
            {
                'model': model.encoder,
                'config': utils.DictWrapper(config_parameters.to_dict()),
            },
            DiskSaver(outputdir),
            n_saved=config_parameters.n_saved,
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=score_function)
        last_checkpoint_saver = Checkpoint(
            {
                'model': model.encoder,
                'config': utils.DictWrapper(config_parameters.to_dict()),
            },
            DiskSaver(outputdir),
            n_saved=1,
            global_step_transform=global_step_from_engine(train_engine))

        decay_steps = len(
            train_dataloader
        ) * epochs if config_parameters.decay_steps is None else config_parameters.decay_steps
        decay_frac = config_parameters.decay_frac

        if config_parameters.use_scheduler:
            scheduler = ignite.handlers.param_scheduler.CosineAnnealingScheduler(
                optimizer, 'lr', optimizer.param_groups[0]['lr'],
                optimizer.param_groups[0]['lr'] * decay_frac, decay_steps)
            warmup_time_in_iters = None
            if config_parameters.warmup_iters is not None:
                warmup_time_in_iters = config_parameters.warmup_iters
            elif config_parameters.warmup_epochs is not None:
                warmup_time_in_iters = len(
                    train_dataloader) * config_parameters.warmup_epochs
            if warmup_time_in_iters is not None:
                logger.info(f"Using warmup with {warmup_time_in_iters} iters")
                scheduler = create_lr_scheduler_with_warmup(
                    scheduler,
                    warmup_start_value=0.0,
                    warmup_duration=warmup_time_in_iters)

            train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        inference_engine.add_event_handler(Events.COMPLETED, checkpoint_saver)
        inference_engine.add_event_handler(Events.COMPLETED,
                                           last_checkpoint_saver)

        @train_engine.on(
            Events.EPOCH_COMPLETED(every=config_parameters.valid_every))
        def valid_eval(train_engine):
            with inference_engine.add_event_handler(Events.COMPLETED,
                                                    log_metrics, "Validation"):
                inference_engine.run(test_dataloader)

        train_engine.run(
            train_dataloader,
            max_epochs=epochs,
            epoch_length=config_parameters.epoch_length,
        )
        output_model = outputdir / checkpoint_saver.last_checkpoint
        if config_parameters.average:
            output_model = outputdir / 'averaged.pt'
            logger.info(f"Averaging best models -> {output_model}")
            averaged_state_dict = utils.average_models(
                [outputdir / f.filename for f in checkpoint_saver._saved])
            torch.save(averaged_state_dict, output_model)

        return output_model


if __name__ == "__main__":
    Fire(RunnerMAE)
