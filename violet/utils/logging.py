from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_writer(log_dir):
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir)
    return writer
