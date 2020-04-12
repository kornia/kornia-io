
import torch
import torchvision

import kornia.io as io


dataset_path: str = "/home/eriba/data/coco"


class TestDaliImageReader:
    '''def test_smoke(self):
        image_reader = io.DaliImageReader()
        assert image_reader is not None

    def test_sanity_types(self):
        image_reader = io.DaliImageReader()
        assert isinstance(image_reader(""), torch.Tensor)'''

    def test_loader(self):
        batch_size=32
        device = torch.device("cuda:0")

        dataset = torchvision.datasets.ImageFolder(
            root=dataset_path, loader=io.DaliImageReader(device))

        loader = torch.utils.data.DataLoader(
            #dataset, batch_size=batch_size, num_workers=8,
            dataset, batch_size=batch_size, num_workers=0,
            collate_fn=io.DaliImageCollateWrapper(batch_size, device),
            pin_memory=False)

        count = 0
        for batch_ndx, (sample, target) in enumerate(loader):
            print(sample.shape)
            print(sample.device)
            print(target.device)
            print(count)
            count += 1
