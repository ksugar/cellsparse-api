import logging
from pathlib import Path
import shutil

from cellsparse.runners import ElephantRunner
from fastapi import APIRouter
import numpy as np
from elephant.common import init_seg_models

from cellsparse_api.utils import (
    decode_image,
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ELEPHANT_MODEL_DIR = str(Path(MODEL_DIR) / "elephant")


@router.post("/elephant/")
async def elephant(body: CellsparseBody):
    runner = ElephantRunner(
        model_dir=str(Path(ELEPHANT_MODEL_DIR) / body.modelname),
        log_path=str(Path(ELEPHANT_MODEL_DIR) / body.modelname / "logs"),
        n_epochs=body.epochs,
        lr=body.lr,
        increment_from=body.modelname,
        crop_size=(body.trainpatch, body.trainpatch),
        n_crops=body.steps,
        min_area=body.minarea,
    )
    img = decode_image(body.b64img)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    lbl = decode_image(body.b64lbl) if body.b64lbl else None
    return run(
        runner,
        body.modelname,
        img,
        lbl,
        body.train,
        body.eval,
        body.simplify_tol,
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
