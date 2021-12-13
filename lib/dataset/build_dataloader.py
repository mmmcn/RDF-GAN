from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from lib.utils.collect import collate_function
from lib.utils.dist_utils import get_dist_info


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_function,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs)

    return data_loader

