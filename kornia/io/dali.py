import numpy as np
import torch


try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as to_pytorch
except ImportError:
    dali = None


if not torch.cuda.is_available():
    raise RuntimeError("DALI requires CUDA support.")


seed = 1549361629


class _DaliImageDecoderPipeline(dali.ops.Pipeline):
    def __init__(self, batch_size: int, num_threads: int, device_id: int):
        super(_DaliImageDecoderPipeline, self).__init__(
            batch_size, num_threads, device_id, seed = seed
        )

        self.input = dali.ops.ExternalSource()
        #self.decode = dali.ops.ImageDecoder(
        #    device='mixed', output_type=dali.types.RGB
        #)
        self.pos_rng_x = dali.ops.Uniform(range = (0.0, 1.0))
        self.pos_rng_y = dali.ops.Uniform(range = (0.0, 1.0))
        self.decode = dali.ops.ImageDecoderCrop(
            device='mixed', output_type=dali.types.RGB, crop=(64, 64))

    @property
    def data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def define_graph(self):
        self.jpegs = self.input()
        #images = self.decode(self.jpegs)
        pos_x = self.pos_rng_x()
        pos_y = self.pos_rng_y()
        images = self.decode(self.jpegs, crop_pos_x=pos_x, crop_pos_y=pos_y)
        return images

    def iter_setup(self):
        images = self.data
        self.feed_input(self.jpegs, images, layout="HWC")


class _DaliImageDecoder:
    def __init__(self, batch_size: int, num_workers: int, device: torch.device) -> None:
        self._pipe = _DaliImageDecoderPipeline(batch_size, num_workers, device)
        self._pipe.build()

        self._device = device

    def __call__(self, input):
        # set data and run the pipeline
        self._pipe.set_data(input)
        out_pipe = self._pipe.run()

        # retrieve dali tensor
        d_images: nvidia.dali.backend_impl.TensorGPU = out_pipe[0].as_tensor()

        # create torch tensor header with expected size
        t_images = torch.empty(
            d_images.shape(), dtype=torch.uint8, device=self._device)

        # populate torch tensor with dali tensor
        to_pytorch.feed_ndarray(d_images, t_images)
        t_images = t_images.permute([0, 3, 1, 2])

        return t_images


class DaliImageCollateWrapper:
    def __init__(self, batch_size: int, device: torch.device):
        self._decoder = _DaliImageDecoder(batch_size, 8, device.index)

        self._device = torch.device("cuda:0")

    def __call__(self, input):
        images = [data[0] for data in input]
        labels = [data[1] for data in input]
  
        t_images = self._decoder(images)
        t_labels = torch.tensor(labels, device=t_images.device)
        return t_images, t_labels


class DaliImageReader:
    def __init__(self, device: torch.device, decode: bool = False) -> None:
        self._loader = _DaliImageDecoder(1, 8, device.index)
        self._decode = decode

    def __call__(self, image_file: str) -> torch.Tensor:
        f = open(image_file, 'rb')
        np_array = np.frombuffer(f.read(), dtype=np.uint8)
        if self._decode:
            return self._decoder([np_array])
        return np_array
