import numpy as np
import torch


try:
    import nvidia.dali as dali
except ImportError:
    dali = None


if not torch.cuda.is_available():
    raise RuntimeError("DALI requires CUDA support.")


'''class _ImageDecoderPipeline(dal.ops.Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(_ImageDecoderPipeline, self).__init__(
            batch_size, num_threads, device_id, seed = seed
        )

        self.input = ops.FileReader(file_root = image_dir)
        self.input_crop_pos = ops.ExternalSource()
        self.input_crop_size = ops.ExternalSource()
        self.input_crop = ops.ExternalSource()
        self.decode = ops.ImageDecoderSlice(device = 'mixed', output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        self.crop_pos = self.input_crop_pos()
        self.crop_size = self.input_crop_size()
        images = self.decode(jpegs, self.crop_pos, self.crop_size)
        return (images, labels)

    def iter_setup(self):
        (crop_pos, crop_size) = pos_size_iter.next()
        self.feed_input(self.crop_pos, crop_pos)
        self.feed_input(self.crop_size, crop_size)'''


class DaliImageReader:
    def __init__(self):
        pass

    def __call__(self, image_path: str) -> torch.Tensor:
        return torch.rand(1)
    

class DaliImageCollateWrapper:
    def __init__(self):
        pass

    def __call__(self, input):
        return torch.rand(1)
