from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gather_knn',
    ext_modules=[
        CUDAExtension('gather_knn_cuda', [
            'csrc/gather_knn.cpp',
            'csrc/gather_knn_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
