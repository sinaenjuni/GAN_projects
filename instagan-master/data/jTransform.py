import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random



class JResize(object):
    def __init__(self, size=(200,200)):
        self.size = size
    def __call__(self, inputs):
        A, B, A_segs, B_segs = inputs
        A = TF.resize(A, self.size, TF.InterpolationMode.BICUBIC)
        B = TF.resize(B, self.size, TF.InterpolationMode.BICUBIC)
        A_segs = TF.resize(A_segs, self.size, TF.InterpolationMode.BICUBIC)
        B_segs = TF.resize(B_segs, self.size, TF.InterpolationMode.BICUBIC)

        # print(A_segs.unique())
        return A, B, A_segs, B_segs


class JRandomCrop(object):
    def __init__(self, size=(256, 256)):
        self.size = size
    def __call__(self, inputs):
        A, B, A_segs, B_segs = inputs

        i, j, h, w = T.RandomCrop.get_params(A, output_size=self.size)
        A = TF.crop(A, i, j, h, w)
        B = TF.crop(B, i, j, h, w)
        A_segs = TF.crop(A_segs, i, j, h, w)
        B_segs = TF.crop(B_segs, i, j, h, w)
        return A, B, A_segs, B_segs


class JRandomHorizontalflip(object):
    def __init__(self, probability = 0.5):
        self.probability = probability

    def __call__(self, inputs):
        A, B, A_segs, B_segs = inputs
        if random.random() >= self.probability:
            A = TF.hflip(A)
            B = TF.hflip(B)
            A_segs = TF.hflip(A_segs)
            B_segs = TF.hflip(B_segs)
        return A, B, A_segs, B_segs


class JToTensor(object):
    def __call__(self, inputs):
        A, B, A_segs, B_segs = inputs
        # A = TF.to_tensor(A)
        # B = TF.to_tensor(B)
        # A_segs = TF.to_tensor(A_segs)
        # B_segs = TF.to_tensor(B_segs)
        return A.div(255), B.div(255), A_segs.div(255), B_segs.div(255)

class JNormalize(object):
    def __call__(self, inputs):
        A, B, A_segs, B_segs = inputs
        A = TF.normalize(A, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        B = TF.normalize(B, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        # A = TF.normalize(A, mean=(0.5), std=(0.5))
        # B = TF.normalize(B, mean=(0.5), std=(0.5))

        # print(A_segs.min(), A_segs.max())
        A_segs = TF.normalize(A_segs, mean=[0.5], std=[0.5])
        B_segs = TF.normalize(B_segs, mean=[0.5], std=[0.5])
        # print(A_segs.min(), A_segs.max())

        return A, B, A_segs, B_segs