import logging
from pathlib import Path
import shutil
from typing import List, Optional

from cellsparse.runners import ElephantRunner
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
import numpy as np
from elephant.common import init_seg_models

from cellsparse_api.utils import (
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    read_image_from_upload_file,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ELEPHANT_MODEL_DIR = str(Path(MODEL_DIR) / "elephant")


def ensure_grayscale(img: np.ndarray):
    if img.ndim != 3:
        return np.mean(img, axis=2)
    return img


@router.post("/elephant/")
async def elephant(
    images: List[UploadFile] = File(...),
    labels: Optional[List[UploadFile]] = File(None),
    json_data: str = Form(...),
):
    try:
        params = StarDistBody.parse_raw(json_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    if params.train and (labels is None or 0 == len(labels)):
        raise HTTPException(status_code=400, detail="Labels are required for training")

    runner = ElephantRunner(
        model_dir=str(Path(ELEPHANT_MODEL_DIR) / body.modelname),
        log_path=str(Path(ELEPHANT_MODEL_DIR) / body.modelname / "logs"),
        n_epochs=params.epochs,
        lr=params.lr,
        increment_from=params.modelname,
        crop_size=(params.trainpatch, params.trainpatch),
        n_crops=params.steps,
        min_area=params.minarea,
    )
    imgs = [await ensure_grayscale(read_image_from_upload_file(img)) for img in images]
    logger.info(f"params: {params}")
    if params.train:
        lbls = [await read_image_from_upload_file(lbl) for lbl in labels]
    else:
        lbls = None
    return run(
        runner,
        params.modelname,
        imgs,
        lbls,
        params.train,
        params.eval,
        params.simplify_tol,
    )


@router.post("/elephant/reset/")
async def elephant_reset(body: CellsparseResetBody):
    p_model_dir = Path(ELEPHANT_MODEL_DIR) / body.modelname
    if p_model_dir.exists() and p_model_dir.is_dir():
        shutil.rmtree(p_model_dir)
    if body.pretrained:
        p_file = p_model_dir / f"unet_{body.modelname}.pth"
        file_name = str(p_file)
        logger.info(f"saving network parameters to {file_name}")
        init_seg_models(file_name, None, "cpu", is_3d=False, url=body.pretrained)
    return ""
