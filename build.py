import os
from os.path import join as pjoin
from setuptools import setup, find_packages

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
import torch
from Cython.Distutils import build_ext
import numpy as np


def find_in_path(name, path):
    "Find a file in a search path"
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    # first check if the CUDAHOME env variable is in use
    if CUDA_HOME is not None:
        home = CUDA_HOME
        nvcc = pjoin(home, 'bin', 'nvcc.exe')
    else:
        print("CUDA_HOME is empty")
        
    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib/x64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    CppExtension(
        "cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'cxx': []},
        include_dirs = [numpy_include]
    )
    ,
    CUDAExtension('gpu_nms',
        ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        extra_compile_args={'cxx': [],
                            'nvcc': ['-arch=sm_61',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options']},
        include_dirs = [numpy_include, CUDA['include']]
    ),
]

setup(    
    name='nms',
    version="1.0",
    packages=['nms'],
    #package_dir={'nms':'nms'},    
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
