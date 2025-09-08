# Installation Guide

The `xsnow` package is designed to install smoothly via `pip`.  
It supports both **standard CPU-based analysis** and **optional GPU acceleration** (via [CuPy](https://cupy.dev/)) for performance-critical workloads.

```{warning}
The `xsnow` package is **not yet published on PyPI**.  
To install it, you currently need to **clone the repository** from GitLab and install locally:

```bash
git clone https://gitlab.com/avacollabra/postprocessing/xsnow
cd xsnow
# pick an install command from below
```


---

## Standard Installation (CPU Only)

For most users, the **standard installation** is sufficient.  
This installs the core package and essential dependencies such as `xarray` and `pandas`. Consider
installing it into a dedicated virtual or conda environment.

```bash
pip install xsnow
```

Once installed, verify it works:

```bash
python -c "import xsnow; print(xsnow.__version__)"
```

---

## Installation with GPU Support

If you have a compatible NVIDIA GPU, `xsnow` can leverage CuPy for substantial speedups in heavy computations.

### Prerequisites

* An NVIDIA GPU with a recent CUDA driver installed
* You do not need the full CUDA toolkit; the driver is enough
* Linux is the most widely tested platform for CuPy prebuilt wheels (Windows/macOS have limited GPU support)

### Install

```bash
pip install xsnow[gpu]
```

This command installs the base package plus a matching CuPy build.
When available, `xsnow` methods will automatically detect and offload work to the GPU.

*Tip*: if no GPU is available, the package gracefully falls back to CPU execution.

---

## Developer Installation

If you want to contribute to `xsnow`, use an editable install.
This way, any local changes to the source code are reflected immediately without reinstallation.

```bash
git clone https://gitlab.com/avacollabra/postprocessing/xsnow
cd xsnow
pip install -e .[gpu,testing,dev]

```

The dev extras include tooling such as test dependencies, jupyter, ipython, matplotlib.
You can now run the test suite to confirm everything works:

```bash
pytest
```
