import logging
from pathlib import Path
import shutil
from typing import List, Optional

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from cellsparse.runners import CellposeRunner
from cellpose import models

from cellsparse_api.utils import (
    decode_image,
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    read_image_from_upload_file,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()


CELLPOSE_MODEL_DIR = str(Path(MODEL_DIR) / "cellpose")


class CellposeBody(CellsparseBody):
    chan1: int = 0
    chan2: int = 0


@router.post("/cellpose/")
async def cellpose(
    images: List[UploadFile] = File(...),
    labels: Optional[List[UploadFile]] = File(None),
    json_data: str = Form(...),
):
    try:
        params = CellposeBody.parse_raw(json_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    if params.train and (labels is None or 0 == len(labels)):
        raise HTTPException(status_code=400, detail="Labels are required for training")

    runner = CellposeRunner(
        channels=[params.chan1, params.chan2],
        save_path=str(Path(CELLPOSE_MODEL_DIR) / params.modelname),
        n_epochs=params.epochs,
        learning_rate=params.lr,
        nimg_per_epoch=params.steps,
        min_area=params.minarea,
    )
    imgs = [await read_image_from_upload_file(img) for img in images]
    logger.info(f"params: {params}")
    if params.train:
        lbls = [await read_image_from_upload_file(lbl) for lbl in labels]
    else:
        lbls = None
    return run(
        runner,
        params.modelname + ".pth",
        imgs,
        lbls,
        params.train,
        params.eval,
        params.simplify_tol,
    )


@router.post("/cellpose/reset/")
async def cellpose_reset(body: CellsparseResetBody):
    p_model_dir = Path(CELLPOSE_MODEL_DIR) / body.modelname
    if p_model_dir.exists() and p_model_dir.is_dir():
        shutil.rmtree(p_model_dir)
    if body.pretrained:
        model = models.CellposeModel(model_type=body.pretrained)
        p_file = p_model_dir / "models" / (body.modelname + ".pth")
        p_file.parent.mkdir(parents=True, exist_ok=True)
        file_name = str(p_file)
        logger.info(f"saving network parameters to {file_name}")
        model.net.save_model(file_name)
    return ""
