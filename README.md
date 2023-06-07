# Cellsparse API

## Install

### Create a Conda environment

```bash
conda create -n cellsparse-api -y python=3.11
```

```bash
conda activate cellsparse-api
```

```bash
conda install -y -c conda-forge cudatoolkit=11.8
```

### Update Pip

```bash
python -m pip install -U pip
```

### Install Cellsparse and dependencies

```bash
python -m pip install git+https://github.com/ksugar/cellsparse-api.git
```

### Work with Tensorflow in Conda

#### Update LD_LIBRARY_PATH

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset CUDNN_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
```

Linux and WSL2 are currently only supported. See details below.

https://www.tensorflow.org/install/pip

If you are using WSL2, `LD_LIBRARY_PATH` will need to be updated as follows.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

#### Update `nvidia-cudnn-cu11`

```bash
python -m pip install --no-deps nvidia-cudnn-cu11==8.6.0.163
```

#### Solve an issue with libdevice

See details [here](https://github.com/tensorflow/tensorflow/issues/58681#issuecomment-1333849966).

```bash
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'unset XLA_FLAGS' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

```bash
conda install -y -c nvidia cuda-nvcc=11.8
```

#### `deactivate` and `activate` the environment

```bash
conda deactivate
conda activate cellsparse-api
```

## Usage

### Launch a server

```bash
uvicorn samapi.main:app
```

The command above will launch a server at http://localhost:8000.

```
INFO:     Started server process [21258]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

For more information, see [uvicorn documentation](https://www.uvicorn.org/#command-line-options).

### Request body

```python
class CellsparseBody(BaseModel):
    modelname: str
    b64img: str
    b64lbl: Optional[str] = None
    train: bool = False
    eval: bool = False
    epochs: Optional[int] = 1
    batchsize: Optional[int] = 8
    steps: Optional[int] = 40
    simplify_tol: Optional[float] = None
```

| key          | value                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------- |
| modelname    | Name of a model for training or inference                                                         |
| b64img       | Base64-encoded image data                                                                         |
| b64lbl       | Base64-encoded label data, required for training                                                  |
| train        | Specify if the request is for training                                                            |
| eval         | Specify if the request is for eval/inference                                                      |
| epochs       | Training epochs                                                                                   |
| batchsize    | Training batch size                                                                               |
| simplify_tol | A parameter to specify how much simplify the output polygons, no simplification happens if `None` |

### Response body

The response body contains a list of [GeoJSON Feature objects](https://geojson.org).

Supporting other formats is a future work.