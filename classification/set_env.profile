# add extra libs
export PYTHONPATH=./libs/:$PYTHONPATH
# disable numpy and scikit learn inner parallism
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
