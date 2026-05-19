from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Dynamic hardware detection for compilation macros
macros = []
sources = ['bindings/python_bindings.cpp', 'src/AOTEngine.cpp', 'src/GraphWrapper.cpp', 'src/Communicator.cpp']

try:
    import torch
    if torch.cuda.is_available():
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            macros.append(("__HIP_PLATFORM_AMD__", "1"))
            print("Detected ROCm environment. Compiling with __HIP_PLATFORM_AMD__")
            mock_hip = False
        else:
            macros.append(("__HIP_PLATFORM_NVIDIA__", "1"))
            print("Detected CUDA environment. Compiling with __HIP_PLATFORM_NVIDIA__")
            mock_hip = False
    else:
        mock_hip = True
except ImportError:
    mock_hip = True

if mock_hip:
    # Forced mock environment via env var or lack of torch/GPU
    macros.append(("MOCK_HIP", "1"))
    sources.extend(['tests/mock_hip.cpp', 'tests/mock_rccl.cpp'])
    print("Warning: No CUDA or ROCm detected. Compiling in MOCK_HIP mode for CPU.")

ext_modules = [
    Extension(
        'python.kinetic_rt._core',
        sources,
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            'include',
            'tests' # for mock_hip.h
        ],
        define_macros=macros,
        language='c++'
    ),
]

# As of Python 3.6, C++11 is the default language standard.
# We explicitly set it to C++14 to ensure compatibility.
def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True

def cpp_flag(compiler):
    flags = ['-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='kinetic_rt',
    version='0.1.0',
    author='Jules',
    description='GraphWrapper using HIP Graphs',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
