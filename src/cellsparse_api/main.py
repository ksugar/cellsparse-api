from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cellsparse_api.routers import (
    cellpose_router,
    elephant_router,
    stardist_router,
)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def handler(request:Request, exc:RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

app.include_router(cellpose_router.router)
app.include_router(elephant_router.router)
app.include_router(stardist_router.router)
