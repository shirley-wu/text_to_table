#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import logging
import math
import os
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq import meters
from fairseq.checkpoint_utils import checkpoint_paths
from fairseq.data import iterators
from fairseq.file_io import PathManager
from fairseq.logging import metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


class Saver:
    def __init__(self):
        self.best = None
        self.keep_best = []

    def save_checkpoint(self, args, trainer, epoch_itr, val_loss):
        # only one worker should attempt to create the required dir
        if args.distributed_rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)

        prev_best = val_loss if self.best is None else self.best
        if val_loss is not None:
            best_function = max if args.maximize_best_checkpoint_metric else min
            self.best = best_function(val_loss, prev_best)

        if args.no_save:
            return

        trainer.consolidate_optimizer()

        if not trainer.is_data_parallel_master:
            return

        def is_better(a, b):
            return a >= b if args.maximize_best_checkpoint_metric else a <= b

        write_timer = meters.StopwatchMeter()
        write_timer.start()

        epoch = epoch_itr.epoch
        end_of_epoch = epoch_itr.end_of_epoch()
        updates = trainer.get_num_updates()

        suffix = getattr(args, "checkpoint_suffix", "")
        checkpoint_conds = collections.OrderedDict()
        save_epoch_checkpoint = (
                end_of_epoch
                and not args.no_epoch_checkpoints
                and epoch % args.save_interval == 0
        )
        checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = save_epoch_checkpoint
        checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
                not save_epoch_checkpoint
                and args.save_interval_updates > 0
                and updates % args.save_interval_updates == 0
        )
        checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
                self.best is None
                or is_better(val_loss, self.best)
        )
        checkpoint_conds[
            "checkpoint_last{}.pt".format(suffix)
        ] = not args.no_last_checkpoints

        extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
        if self.best is not None:
            extra_state.update({"best": self.best})

        if args.keep_best_checkpoints > 0 and (len(self.keep_best) < args.keep_best_checkpoints or (
                val_loss is not None and not is_better(self.keep_best[-1][0], val_loss))):
            ckpt_name = "checkpoint{}{}.best_{:.4f}.pt".format(epoch, suffix, val_loss) if save_epoch_checkpoint \
                else "checkpoint_{}_{}{}.best_{:.4f}.pt".format(epoch, updates, suffix, val_loss)
            checkpoint_conds[ckpt_name] = True
            self.keep_best.append((val_loss, ckpt_name))
            self.keep_best = sorted(self.keep_best)

        checkpoints = [
            os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
        ]
        if len(checkpoints) > 0:
            trainer.save_checkpoint(checkpoints[0], extra_state)
            for cp in checkpoints[1:]:
                PathManager.copy(checkpoints[0], cp, overwrite=True)

            write_timer.stop()
            logger.info(
                "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                    checkpoints[0], epoch, updates, val_loss, write_timer.sum
                )
            )

        if not end_of_epoch and args.keep_interval_updates > 0:
            # remove old checkpoints; checkpoints are sorted in descending order
            checkpoints = checkpoint_paths(
                args.save_dir, pattern=r"checkpoint_\d+_(\d+)\.pt"
            )
            for old_chk in checkpoints[args.keep_interval_updates:]:
                if os.path.lexists(old_chk):
                    os.remove(old_chk)

        if args.keep_last_epochs > 0:
            # remove old epoch checkpoints; checkpoints are sorted in descending order
            checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
            for old_chk in checkpoints[args.keep_last_epochs:]:
                if os.path.lexists(old_chk):
                    os.remove(old_chk)

        if len(self.keep_best) > args.keep_best_checkpoints:
            for _, x in self.keep_best[args.keep_best_checkpoints:]:
                x = os.path.join(args.save_dir, x)
                if os.path.lexists(x):
                    os.remove(x)
            self.keep_best = self.keep_best[:args.keep_best_checkpoints]


def main(args):
    saver = Saver()
    utils.import_user_module(args)

    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.batch_size
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr, saver)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr, saver):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_losses = [None]
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, saver
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch, saver):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
            (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
            or num_updates >= max_update
            or (
                    args.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % args.save_interval_updates == 0
                    and num_updates >= args.validate_after_updates
            )
    )
    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
                          or num_updates >= max_update
                          or (
                                  args.validate_interval_updates > 0
                                  and num_updates > 0
                                  and num_updates % args.validate_interval_updates == 0
                          )
                  ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, saver)

    # Stopping conditions
    should_stop = (
            should_stop_early(args, valid_losses[0])
            or num_updates >= max_update
            or (
                    args.stop_time_hours > 0
                    and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
            )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        saver.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets, saver):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values(), saver)
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats, saver):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(saver.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            saver.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
