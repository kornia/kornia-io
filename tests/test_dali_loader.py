
import torch
import torchvision

import kornia.io as io


dataset_path: str = "/home/eriba/data/coco"


class TestDaliImageReader:
    def test_smoke(self):
        image_reader = io.DaliImageReader()
        assert image_reader is not None

    def test_sanity_types(self):
        image_reader = io.DaliImageReader()
        assert isinstance(image_reader(""), torch.Tensor)

    def test_loader(self):

        dataset = torchvision.datasets.ImageFolder(
            root=dataset_path, loader=io.DaliImageReader())

        loader = torch.utils.data.DataLoader(
            #dataset, batch_size=2, collate_fn=io.DaliImageCollateWrapper(),
            dataset, batch_size=2,
            pin_memory=True)

        for batch_ndx, sample in enumerate(loader):
            #import pdb;pdb.set_trace()
            print(sample[0][0].shape)
            print(sample[0][0].is_pinned())
