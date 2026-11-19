#!/bin/bash
#
# Example SLURM environment setup (from the cluster used for the paper).
# Copy to prerequisites.sh and adapt; the submit_slurm_array.sh scripts
# source ./prerequisites.sh automatically if it exists.
#
# EDIT THE BLOCK MARKED "cluster-specific" for your site; everything else is derived.

# ---------------------------------------------------------------- cluster-specific
module purge

# Note that the modules loaded on the login node differ from compute nodes.
# Use `module avail` on the login node to see modules that can be loaded to fix errors.
#module load autotools prun/2.2 gnu9/9.4.0 hwloc/2.5.0 ucx/1.11.2 libfabric/1.13.0 openmpi4/4.1.1 ohpc
module load gnu12/12.2.0

module load \
    linux-rocky8-x86_64/gcc/12.2.0/python/3.11.1-efo4f5x \
    linux-rocky8-x86_64/gcc/12.2.0/swig/4.1.1-qarng36 \
    linux-rocky8-x86_64/gcc/12.2.0/openblas/0.3.21-zy2y7i4 \
    linux-rocky8-x86_64/gcc/12.2.0/hdf5/1.12.1-cdxzvhd \
    linux-rocky8-x86_64/gcc/12.2.0/boost/1.81.0-5k3pf6l

# linux-rocky8-x86_64/gcc/12.2.0/boost/1.82.0-7gfxh6x

# Other optional modules -- can remove if undesired
module load \
    linux-rocky8-x86_64/gcc/12.2.0/git/2.39.1-66kunjv \
    linux-rocky8-x86_64/gcc/12.2.0/vim/9.0.0045-rtvvtu6

# Might be that not all of these are required to install and use AMICI.
PATH_TO_OPENBLAS="/opt/ohpc/pub/spack/develop/opt/spack/linux-rocky8-zen2/gcc-12.2.0/openblas-0.3.21-zy2y7i4qybyoxlh6itcvjgiruhsimi7b"
export BLAS_INCDIR="$PATH_TO_OPENBLAS/include/"
export BLAS_CFLAGS="-I$BLAS_INCDIR"
export BLAS_LIBS="-lopenblas"
export HDF5_BASE="/opt/ohpc/pub/spack/develop/opt/spack/linux-rocky8-zen2/gcc-12.2.0/hdf5-1.12.1-cdxzvhdnb2hdcb4ib3aoifzqikbjzly3"
export LDFLAGS="-L$PATH_TO_OPENBLAS/lib/${LDFLAGS:+ $LDFLAGS}"
export LD_LIBRARY_PATH="$HDF5_BASE/lib:$PATH_TO_OPENBLAS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------- venv
# Activate the virtual environment that holds requirements.txt
# after the module loads above, so its python/pip shadow the module ones.
# Override with VENV_DIR=/path/to/venv.
PREREQ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PREREQ_DIR/.venv}"
source "$VENV_DIR/bin/activate"
echo "DEBUG: python3=$(which python3)"
echo "DEBUG: VENV = $VIRTUAL_ENV"
python3 -c "import yaml2sbml"
