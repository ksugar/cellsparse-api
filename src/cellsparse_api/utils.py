import base64
import io
import logging
import os
from typing import (
    Optional,
    Tuple,
)
from csbdeep.utils import normalize
from geojson import Feature
from geojson import Polygon as geojson_polygon
import numpy as np
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import Polygon as shapely_polygon
from skimage import measure

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())
logger = logging.getLogger("uvicorn")

try:
    Image.MAX_IMAGE_PIXELS = int(
        os.getenv("PIL_MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    )
except:
    logger.warning(
        "PIL.Image.MAX_IMAGE_PIXELS is set to None, potentially exposing the system to decompression bomb attacks."
    )
    Image.MAX_IMAGE_PIXELS = None


MODEL_DIR = os.environ.get(
    "CELLSPARSE_MODEL_DIR",
    str(Path.home() / ".cellsparse/models"),
)


def decode_image(b64data: str):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64data))))


def mask_to_geometry(
    mask: np.ndarray,
    downsample: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
    simplify_tol=None,
):
    # modified from https://github.com/MouseLand/cellpose_web/blob/main/utils.py
    mask = np.pad(mask, 1)  # handle edges properly by zero-padding
    contours_find = measure.find_contours(mask, 0.5)
    if len(contours_find) == 1:
        index = 0
    else:
        pixels = []
        for _, item in enumerate(contours_find):
            pixels.append(len(item))
        index = np.argmax(pixels)
    contour = contours_find[index]
    contour -= 1  # reset padding
    contour_as_numpy = contour[:, np.argsort([1, 0])]
    contour_as_numpy *= downsample
    contour_as_numpy[:, 0] += offset[0]
    contour_as_numpy[:, 1] += offset[1]
    contour_asList = contour_as_numpy.tolist()
    if simplify_tol is not None and 0 < simplify_tol:
        poly_shapely = shapely_polygon(contour_asList)
        poly_shapely_simple = poly_shapely.simplify(
            simplify_tol, preserve_topology=False
        )
        contour_asList = list(poly_shapely_simple.exterior.coords)
    return geojson_polygon([contour_asList])


def postprocess(pred, simplify_tol=None):
    index_number = 0
    features = []
    for value in np.unique(pred):
        if value == 0:
            continue
        features.append(
            Feature(
                geometry=mask_to_geometry(
                    (pred == value).astype(np.uint8),
                    simplify_tol=simplify_tol,
                ),
                properties={"object_idx": index_number, "label": "object"},
            )
        )
        index_number += 1
    return features


def run(
    runner,
    modelname,
    img,
    lbl=None,
    train=False,
    eval=False,
    simplify_tol=None,
):
    img = normalize(img, 0, 100, axis=(0, 1))
    if train:
        lbl = lbl.astype(int) - 1
        runner._train([img], [lbl], [img], [lbl], modelname)
    if eval:
        pred = runner._eval([img], modelname)[0]
        return postprocess(pred, simplify_tol)
    return []


class CellsparseBody(BaseModel):
    modelname: str
    b64img: str
    b64lbl: Optional[str] = None
    train: bool = False
    eval: bool = False
    epochs: int = 1
    trainpatch: int = 224
    batchsize: int = 8
    steps: int = 200
    lr: float = 0.001
    minarea: float = 10.0
    simplify_tol: float = None


class CellsparseResetBody(BaseModel):
    modelname: str
    pretrained: Optional[str] = None
