import os
import glob
import torch
import shutil

import numpy as np
import torch.nn as nn

from logger_config import logger


class AttrDict:
    pass


def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    logger.info('Wrote checkpoint: %s', filename)
    ck_dir = os.path.dirname(filename)
    # Always update model_last.mdl
    try:
        last_path = os.path.join(ck_dir, 'model_last.mdl')
        shutil.copyfile(filename, last_path)
        logger.info('Updated model_last.mdl -> %s', last_path)
    except Exception as exc:
        logger.warning('Failed to update model_last.mdl: %s', exc)

    # Update model_best.mdl when this is the best checkpoint
    if is_best:
        try:
            best_path = os.path.join(ck_dir, 'model_best.mdl')
            shutil.copyfile(filename, best_path)
            logger.info('Updated model_best.mdl -> %s', best_path)
        except Exception as exc:
            logger.warning('Failed to update model_best.mdl: %s', exc)


def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
            logger.info('{}: {}'.format(name, np.prod(list(p.size()))))

    logger.info('Number of parameters: {}M'.format(num_parameters // 10**6))
    return num_parameters


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def call_model_forward(model, batch_dict):
    """
    Properly calls model.forward() with batch_dict, handling DataParallel compatibility.
    Extracts required positional arguments explicitly to avoid issues with DataParallel kwargs unpacking.
    
    Args:
        model: The model instance (may be wrapped with DataParallel)
        batch_dict: Dictionary containing batch data with keys: hr_token_ids, hr_mask, hr_token_type_ids,
                   tail_token_ids, tail_mask, tail_token_type_ids, head_token_ids, head_mask, head_token_type_ids
    
    Returns:
        Model output dictionary
    """
    return model(
        hr_token_ids=batch_dict['hr_token_ids'],
        hr_mask=batch_dict['hr_mask'],
        hr_token_type_ids=batch_dict['hr_token_type_ids'],
        tail_token_ids=batch_dict['tail_token_ids'],
        tail_mask=batch_dict['tail_mask'],
        tail_token_type_ids=batch_dict['tail_token_type_ids'],
        head_token_ids=batch_dict['head_token_ids'],
        head_mask=batch_dict['head_mask'],
        head_token_type_ids=batch_dict['head_token_type_ids'],
        only_ent_embedding=batch_dict.get('only_ent_embedding', False)
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
