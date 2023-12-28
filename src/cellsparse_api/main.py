from fastapi import FastAPI

from cellsparse_api.routers import (
    cellpose_router,
    elephant_router,
    stardist_router,
)

app = FastAPI()

app.include_router(cellpose_router.router)
app.include_router(elephant_router.router)
app.include_router(stardist_router.router)
