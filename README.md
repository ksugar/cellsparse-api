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