# Cellsparse API

A web API for [Cellsparse](https://github.com/ksugar/cellsparse-core) implemented with [FastAPI](https://fastapi.tiangolo.com).

[qupath-extension-cellsparse](https://github.com/ksugar/qupath-extension-cellsparse) implements a client for [QuPath](https://qupath.github.io).

This is a part of the following paper. Please [cite](#citation) it when you use this project.

- Sugawara, K. [*Training deep learning models for cell image segmentation with sparse annotations.*](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1) bioRxiv 2023. doi:10.1101/2023.06.13.544786

## Install

### Mac OSX (Apple silicon or AMD GPUs)

Please check that your system satisfies the requirements listed in [the official instructions](https://developer.apple.com/metal/tensorflow-plugin/).\
Othrewise, please follow [the installation steps for the CPU version](./#windows-native-mac-osx-linux-wsl2-cpu).

```bash
conda create -n cellsparse-api -y python=3.11
conda activate cellsparse-api
python -m pip install -U pip
python -m pip install "cellsparse-api[tensorflow-macos] @ git+https://github.com/ksugar/cellsparse-api.git"
```

### Windows Native with CUDA-compatible GPU

Microsoft Visual C++ 14.0 or greater is required.  
Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

```bash
conda create -n cellsparse-api -y python=3.10
conda activate cellsparse-api
python -m pip install -U pip
conda install -y -c conda-forge cudatoolkit=11.3 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
python -m pip install git+https://github.com/ksugar/stardist-sparse.git
set PYTHONUTF8=1
python -m pip install git+https://github.com/ksugar/cellsparse-api.git
set PYTHONUTF8=0
python -m pip uninstall -y torch torchvision
python -m pip install --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu113
```

### Windows Native, Mac OSX, Linux, WSL2 (CPU)

Please note that training with CPU is very slow.

On Windows Native, Microsoft Visual C++ 14.0 or greater is required.  
Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

```bash
conda create -n cellsparse-api -y python=3.11
conda activate cellsparse-api
python -m pip install -U pip
python -m pip install "cellsparse-api[tensorflow] @ git+https://github.com/ksugar/cellsparse-api.git"
```

### Linux or WSL2 with CUDA-compatible GPU

```bash
conda create -n cellsparse-api -y python=3.11
conda activate cellsparse-api
python -m pip install -U pip
conda install -y -c conda-forge cudatoolkit=11.8
python -m pip install "cellsparse-api[tensorflow] @ git+https://github.com/ksugar/cellsparse-api.git"
```

The following steps are required only if you're using Linux or WSL2 with CUDA-compatible GPU.  
If it is not the case, you can move to the [Usage](#usage) section.

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
uvicorn cellsparse_api.main:app
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
    epochs: int = 1
    trainpatch: int = 224
    batchsize: int = 8
    steps: int = 200
    lr: float = 0.001
    minarea: float = 10.0
    simplify_tol: float = None
```

| key          | value                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------- |
| modelname    | Name of a model for training or inference                                                         |
| b64img       | Base64-encoded image data                                                                         |
| b64lbl       | Base64-encoded label data, required for training                                                  |
| train        | Specify if the request is for training                                                            |
| eval         | Specify if the request is for eval/inference                                                      |
| epochs       | Training epochs                                                                                   |
| trainpatch   | Training patch size, Cellpose does not support this parameter                                     |
| batchsize    | Training batch size                                                                               |
| steps        | Training steps per epoch                                                                          |
| lr           | Training learning rate                                                                            |
| minarea      | Objects smaller than this value are removed in post processing                                    |
| simplify_tol | A parameter to specify how much simplify the output polygons, no simplification happens if `None` |

### Response body

The response body contains a list of [GeoJSON Feature objects](https://geojson.org).

Supporting other formats is a future work.

## Citation

Please cite my paper on [bioRxiv](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1).

```.bib
@article {Sugawara2023.06.13.544786,
	author = {Ko Sugawara},
	title = {Training deep learning models for cell image segmentation with sparse annotations},
	elocation-id = {2023.06.13.544786},
	year = {2023},
	doi = {10.1101/2023.06.13.544786},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning is becoming more prominent in cell image analysis. However, collecting the annotated data required to train efficient deep-learning models remains a major obstacle. I demonstrate that functional performance can be achieved even with sparsely annotated data. Furthermore, I show that the selection of sparse cell annotations significantly impacts performance. I modified Cellpose and StarDist to enable training with sparsely annotated data and evaluated them in conjunction with ELEPHANT, a cell tracking algorithm that internally uses U-Net based cell segmentation. These results illustrate that sparse annotation is a generally effective strategy in deep learning-based cell image segmentation. Finally, I demonstrate that with the help of the Segment Anything Model (SAM), it is feasible to build an effective deep learning model of cell image segmentation from scratch just in a few minutes.Competing Interest StatementKS is employed part-time by LPIXEL Inc.},
	URL = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786},
	eprint = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786.full.pdf},
	journal = {bioRxiv}
}
```
