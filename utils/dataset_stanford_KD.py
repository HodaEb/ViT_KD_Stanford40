import logging
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from PIL import Image
import torch
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset, DatasetFolder
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data import Dataset

from torchvision.datasets.folder import make_dataset, default_loader

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class stanford40(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            transform2: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.transform2 = transform2

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample1 = self.transform(sample)
            sample2 = self.transform2(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target

def get_loader_KD(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    teacher_transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    teacher_transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # student_transform_train = transforms.Compose([
    #     transforms.Resize((400, 400)),
    #     transforms.RandomCrop((300, 300), padding=4),
    #     # transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     # transforms.CenterCrop(300),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    #     ),
    #     transforms.RandomGrayscale(0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # student_transform_test = transforms.Compose([
    #     transforms.Resize((300, 300)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    student_transform_train = transforms.Compose([
    # transforms.RandomResizedCrop((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomApply([
    #   transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    # ),
    # transforms.RandomGrayscale(0.2),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # transforms.Resize((224, 224)),
    # transforms.CenterCrop((224, 224)),

    # transforms.Resize((380, 380)),
    # transforms.RandomCrop((320, 320)),

    transforms.Resize((320, 320)),
    transforms.CenterCrop((320, 320)),

    transforms.RandomResizedCrop(size=(320, 320), scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(degrees =(0, 10)),
    transforms.RandomRotation(degrees =(0, 23)),
    transforms.RandomApply([
      transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    # transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    student_transform_test = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # transforms.Resize((224, 224)),
        # transforms.CenterCrop((224, 224)),

        transforms.Resize((320, 320)),
        transforms.CenterCrop((320, 320)),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.dataset == "cifar10":
        raise ValueError('{} doesnt support the KD'.format(args.dataset))
    elif args.dataset == "stanford40":

        trainset = stanford40(root='/content/Standford40/StanfordActionDataset/train',
                                        transform=teacher_transform_train, transform2=student_transform_train)

        testset = stanford40(root='/content/Standford40/StanfordActionDataset/test',
                                       transform=teacher_transform_test, transform2=student_transform_test)

    else:
        raise ValueError('{} doesnt support the KD'.format(args.dataset))

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
