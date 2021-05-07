from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='AFSD',
        version='1.0',
        description='Learning Salient Boundary Feature for Anchor-free '
                    'Temporal Action Localization',
        author='Chuming Lin, Chengming Xu',
        author_email='chuminglin@tencent.com, cmxu18@fudan.edu.cn',
        packages=find_packages(
            exclude=('configs', 'models', 'output', 'datasets')
        ),
        ext_modules=[
            CUDAExtension('boundary_max_pooling_cuda', [
                'AFSD/prop_pooling/boundary_max_pooling_cuda.cpp',
                'AFSD/prop_pooling/boundary_max_pooling_kernel.cu'
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
