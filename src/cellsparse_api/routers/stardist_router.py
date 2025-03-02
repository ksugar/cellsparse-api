import logging
from pathlib import Path
import shutil
from typing import List, Optional

from cellsparse.runners import StarDistRunner
from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from stardist.models import StarDist2D

from cellsparse_api.utils import (
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    read_image_from_upload_file,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()

STARDIST_BASE_DIR = str(Path(MODEL_DIR) / "stardist")


class StarDistBody(CellsparseBody):
    n_channels_in: int = 1


@router.post("/stardist/")
async def stardist(
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

    runner = StarDistRunner(
        n_channel_in=params.n_channels_in,
        grid=(2, 2),
        basedir=STARDIST_BASE_DIR,
        use_gpu=False,
        train_epochs=params.epochs,
        train_patch_size=(params.trainpatch, params.trainpatch),
        train_batch_size=params.batchsize,
        train_steps_per_epoch=params.steps,
        min_area=params.minarea,
        train_learning_rate=params.lr,
    )
    imgs = [await read_image_from_upload_file(img) for img in images]
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


@router.post("/stardist/reset/")
async def stardist_reset(body: CellsparseResetBody):
    p_model_dir = Path(STARDIST_BASE_DIR) / body.modelname
    if p_model_dir.exists() and p_model_dir.is_dir():
        shutil.rmtree(p_model_dir)
    if body.pretrained:
        model = StarDist2D.from_pretrained(body.pretrained)
        p_model_dir.mkdir(parents=True, exist_ok=True)
        p_file = p_model_dir / "weights_last.h5"
        file_name = str(p_file)
        logger.info(f"saving network parameters to {file_name}")
        model.keras_model.save_weights(file_name)
    return ""
