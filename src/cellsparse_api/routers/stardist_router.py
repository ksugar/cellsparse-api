import logging
from pathlib import Path
import shutil

from cellsparse.runners import StarDistRunner
from fastapi import APIRouter
from stardist.models import StarDist2D

from cellsparse_api.utils import (
    decode_image,
    CellsparseBody,
    CellsparseResetBody,
    MODEL_DIR,
    run,
)

logger = logging.getLogger(__name__)

router = APIRouter()

STARDIST_BASE_DIR = str(Path(MODEL_DIR) / "stardist")


class StarDistBody(CellsparseBody):
    n_channels_in: int = 1


@router.post("/stardist/")
async def stardist(body: StarDistBody):
    runner = StarDistRunner(
        n_channel_in=body.n_channels_in,
        grid=(2, 2),
        basedir=STARDIST_BASE_DIR,
        use_gpu=False,
        train_epochs=body.epochs,
        train_patch_size=(body.trainpatch, body.trainpatch),
        train_batch_size=body.batchsize,
        train_steps_per_epoch=body.steps,
        min_area=body.minarea,
        train_learning_rate=body.lr,
    )
    img = decode_image(body.b64img)
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
