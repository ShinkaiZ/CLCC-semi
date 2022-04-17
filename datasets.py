import cv2
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.sampler import Sampler
import itertools

warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    refer to https://github.com/PRIS-CV/DCRNet/blob/master/utils/dataloader.py
    """

    def __init__(self, cfg, roots, mode='train', trainsize=320, scale=(0.99, 1.01)):
        self.cfg = cfg
        self.trainsize = trainsize
        self.scale = scale
        self.mode = mode
        self.images = []
        self.gts = []
        self.dataset_lens = []
        for root in roots:
            if mode == 'train':
                self.data_root = os.path.join(root, 'TrainDataset')
                self.transform = self.get_augmentation()
            elif mode == 'val':
                self.data_root = os.path.join(root, 'ValidationDataset')
                self.transform = A.Compose(
                    [A.Resize(trainsize, trainsize), ])
            elif mode == 'test':
                self.data_root = os.path.join(root, 'TestDataset')
                self.transform = A.Compose([A.Resize(trainsize, trainsize), ])
            else:
                raise KeyError('MODE ERROR')
            image_root = os.path.join(self.data_root, 'images')
            gt_root = os.path.join(self.data_root, 'masks')
            _images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if
                              f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
            _gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if
                           f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
            self.images += _images
            self.gts += _gts
            self.dataset_lens.append(len(self.images))
        self.filter_files()
        self.size = len(self.images)
        self.to_tensors = A.Compose([A.Normalize(), ToTensorV2()])

    def __len__(self):
        return self.size

    def lens(self):
        return self.dataset_lens

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_3 = cv2.imread(self.gts[index])
        if self.cfg.DATA.SEG_CLASSES == 1:
            mask = mask_3.sum(axis=2) // 384
        else:
            mask = mask_3 // 128
        mask = mask.astype(np.uint8)
        assert mask.max() == 1 or mask.max() == 0
        data_np = self.transform(image=image, mask=mask)
        data_t = self.to_tensors(image=data_np['image'], mask=data_np['mask'])
        data = {'imidx': index, 'path': self.images[index], 'image': data_t['image'], 'label': data_t['mask']}

        return data

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        for img_path, gt_path in zip(self.images, self.gts):
            img = cv2.imread(img_path)
            gt = cv2.imread(gt_path)
            assert gt.max() == 255
            assert gt.min() == 0
            assert img.shape == gt.shape
            assert img_path.split('/')[-1].split('.')[0].split('_')[0] == \
                   gt_path.split('/')[-1].split('.')[0].split('_')[0], (img_path, gt_path)

    def get_augmentation(self):
        return A.Compose([
            A.Resize(self.trainsize, self.trainsize),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ])


def worker_init_fn(worker_id):
    random.seed(worker_id)


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, cfg, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.cfg = cfg
        self.primary_indices = primary_indices  # * self.cfg.DATA.REPEAT
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        # print("len: ", len(self.primary_indices), len(self.secondary_indices))

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size >= 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        if len(self.secondary_indices) == 0 and self.secondary_batch_size == 0:
            return (
                primary_batch + primary_batch
                for (primary_batch, primary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                       grouper(primary_iter, self.primary_batch_size))
            )
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def get_dataset(mode, cfg, trainsize=320, scale=(0.75, 1)):
    # print('===mode===>', mode)
    data_root = []
    if "Kvasir" in cfg.DATA.NAME:
        data_root.append(os.path.join(cfg.DIRS.DATA, 'Kvasir-SEG'))
    if "ISIC" in cfg.DATA.NAME:
        data_root.append(os.path.join(cfg.DIRS.DATA, 'ISIC'))
    if mode == 'train':
        dts = ImageDataset(cfg=cfg, roots=data_root, mode='train', trainsize=trainsize, scale=scale)
        batch_size = cfg.TRAIN.BATCH_SIZE

        total_slices = len(dts)
        # print('==> total_slices', total_slices)
        labeled_slice_len = int(total_slices * cfg.DATA.LABEL)
        idxs = list(range(total_slices))
        fold_len = int(cfg.DATA.LABEL * total_slices)
        labeled_idxs = idxs[fold_len * cfg.TRAIN.FOLD: fold_len * (cfg.TRAIN.FOLD + 1)]
        unlabeled_idxs = list(set(idxs) - set(labeled_idxs))

        assert len(labeled_idxs) == labeled_slice_len

        # print('==> labeled indexes', labeled_idxs)
        # print('==> unlabeled indexes', unlabeled_idxs)
        batch_sampler = TwoStreamBatchSampler(cfg,
                                              labeled_idxs, unlabeled_idxs, batch_size,
                                              batch_size - cfg.TRAIN.LB_BATCH_SIZE)
        dataloader = DataLoader(dts, batch_sampler=batch_sampler,
                                num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init_fn)
    elif mode == 'valid':
        dts = ImageDataset(cfg=cfg, roots=data_root, mode='val', trainsize=trainsize, scale=scale)
        batch_size = cfg.VAL.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)
    elif mode == 'test':
        dts = ImageDataset(cfg=cfg, roots=data_root, mode='test', trainsize=trainsize, scale=scale)
        batch_size = cfg.TEST.BATCH_SIZE
        dataloader = DataLoader(dts, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=cfg.SYSTEM.NUM_WORKERS)

    else:
        raise KeyError(f"mode error: {mode}")
    return dataloader
