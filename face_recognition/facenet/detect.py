import numpy as np
from PIL import Image
from algo.facenet import Facenet


class Detect:
    def __init__(self):
        super().__init__()
        self.face_net = Facenet()

    def __call__(self, img):
        if type(img) == str:
            img = Image.open(img)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.face_net.output_feature(img)


def extract_image_from_facenet(img: Image):
    detect = Detect()
    feature = detect(img)
    return feature