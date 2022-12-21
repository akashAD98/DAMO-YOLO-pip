#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import glob
import re
from os import path
import io
import setuptools
import torch
from torch.utils.cpp_extension import CppExtension

torch_ver = [int(x) for x in torch.__version__.split('.')[:2]]
assert torch_ver >= [1, 7], 'Requires PyTorch >= 1.7'


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "yolov7", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, 'damo', 'layers', 'csrc')

    main_source = path.join(extensions_dir, 'vision.cpp')
    sources = glob.glob(path.join(extensions_dir, '**', '*.cpp'))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            'damo._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


with open('damo/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(),
                        re.MULTILINE).group(1)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='damo_yolo',
    version=version,
    author='Akash_Desai',
    
    
    license="MIT",
    description="Packaged version of the DAMO repository",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/akashAD98/DAMO-YOLO-pip",
    packages=setuptools.find_packages(),
    
    python_requires='>=3.6',
    long_description=long_description,
    ext_modules=get_extensions(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
)

