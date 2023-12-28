import logging
from pathlib import Path
import shutil

from fastapi import APIRouter
from cellsparse.runners import CellposeRunner
from cellpose import models

from cellsparse_api.utils import (
    decode_image,
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()


CELLPOSE_MODEL_DIR = str(Path(MODEL_DIR) / "cellpose")


class CellposeBody(CellsparseBody):
    chan1: int = 0
    chan2: int = 0


@router.post("/cellpose/")
async def cellpose(body: CellposeBody):
    runner = CellposeRunner(
        channels=[body.chan1, body.chan2],
        save_path=str(Path(CELLPOSE_MODEL_DIR) / body.modelname),
        n_epochs=body.epochs,
        learning_rate=body.lr,
        nimg_per_epoch=body.steps,
        min_area=body.minarea,
    )
    img = decode_image(body.b64img)
    lbl = decode_image(body.b64lbl) if body.b64lbl else None
    return run(
        runner,
        body.modelname + ".pth",
        img,
        lbl,
        body.train,
        body.eval,
        body.simplify_tol,
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
