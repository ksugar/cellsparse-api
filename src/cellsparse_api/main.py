import base64
import io
from typing import (
    Optional,
    Tuple,
)

from cellsparse.runners import (
    CellposeRunner,
    ElephantRunner,
    StarDistRunner,
)
from csbdeep.utils import normalize
from fastapi import FastAPI
from geojson import Feature
from geojson import Polygon as geojson_polygon
import numpy as np
from PIL import Image
from pydantic import BaseModel
from shapely.geometry import Polygon as shapely_polygon
from skimage import measure

app = FastAPI()


def decode_image(b64data: str):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64data))))


def run(
    runner,
    modelname,
    b64img,
    b64lbl=None,
    train=False,
    eval=False,
    simplify_tol=None,
):
    img = normalize(decode_image(b64img), 0, 100, axis=(0, 1))
    if train:
        lbl = decode_image(b64lbl).astype(int) - 1
        runner._train([img], [lbl], [img], [lbl], modelname)
    if eval:
        pred = runner._eval([img], modelname)[0]
        return postprocess(pred, simplify_tol)
    return []


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
    if simplify_tol is not None:
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


class CellsparseBody(BaseModel):
    modelname: str
    b64img: str
    b64lbl: Optional[str] = None
    train: bool = False
    eval: bool = False
    epochs: int = 1
    batchsize: int = 8
    steps: int = 40
    simplify_tol: float = None


STARDIST_BASE_DIR = "./models/stardist"
STARDIST_PATCH_SIZE = (224, 224)


@app.post("/stardist/")
async def stardist(body: CellsparseBody):
    runner = StarDistRunner(
        grid=(2, 2),
        basedir=STARDIST_BASE_DIR,
        use_gpu=False,
        train_epochs=1,
        train_patch_size=STARDIST_PATCH_SIZE,
        train_batch_size=8,
        train_steps_per_epoch=200,
        min_area=10,
        train_learning_rate=0.001,
    )
    return run(
        runner,
        body.modelname,
        body.b64img,
        body.b64lbl,
        body.train,
        body.eval,
        body.simplify_tol,
    )


CELLPOSE_MODEL_DIR = "./models/cellpose"


@app.post("/cellpose/")
async def cellpose(body: CellsparseBody):
    runner = CellposeRunner(
        save_path=CELLPOSE_MODEL_DIR,
        n_epochs=20,
        nimg_per_epoch=40,
        min_area=10,
    )
    return run(
        runner,
        body.modelname + ".pth",
        body.b64img,
        body.b64lbl,
        body.train,
        body.eval,
        body.simplify_tol,
    )


ELEPHANT_MODEL_DIR = "./models/elephant"
ELEPHANT_LOG_PATH = "./models/elephant/logs"


@app.post("/elephant/")
async def elephant(body: CellsparseBody):
    runner = ElephantRunner(
        model_dir=ELEPHANT_MODEL_DIR,
        log_path=ELEPHANT_LOG_PATH,
        n_epochs=1,
        lr=0.001,
        r_min=3,
        increment_from=body.modelname,
        crop_size=(128, 128),
        n_crops=200,
        min_area=10,
    )
    return run(
        runner,
        body.modelname,
        body.b64img,
        body.b64lbl,
        body.train,
        body.eval,
        body.simplify_tol,
    )
