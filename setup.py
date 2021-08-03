import pathlib

from setuptools import setup
try:
    from setuptools_rust import Binding, RustExtension
except ImportError:
    import sys
    import subprocess

    subprocess.call([sys.executable, '-m', 'pip', 'install',
                     'setuptools-rust'])
    from setuptools_rust import Binding, RustExtension

STUBS_DIR = "retworkx-stubs"

def readme():
    with open('README.md') as f:
        return f.read()

def get_stub_files():
    current_dir = pathlib.Path(__file__).parents[0]
    pyi_files = [
        str(pyi_file.relative_to(current_dir)) for pyi_file in current_dir.glob(f"{STUBS_DIR}/**/*.pyi")
    ]
    py_typed_files = [
        str(typed_file.relative_to(current_dir)) for typed_file in current_dir.glob(f"{STUBS_DIR}/**/py.typed")
    ]
    return pyi_files + py_typed_files


mpl_extras = ['matplotlib>=3.0']
graphviz_extras = ['pydot>=1.4', 'pillow>=5.4']


setup(
    name="retworkx",
    version="0.10.0",
    description="A python graph library implemented in Rust",
    long_description=readme(),
    long_description_content_type='text/markdown',
    author="Matthew Treinish",
    author_email="mtreinish@kortar.org",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="Networks network graph Graph Theory DAG",
    url="https://github.com/Qiskit/retworkx",
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/retworkx/issues",
        "Source Code": "https://github.com/Qiskit/retworkx",
        "Documentation": "https://qiskit.org/documentation/retworkx",
    },
    rust_extensions=[RustExtension("retworkx.retworkx", "Cargo.toml",
                                   binding=Binding.PyO3)],
    include_package_data=True,
    packages=["retworkx", "retworkx.visualization", "retworkx-stubs"],
    data_files=[("", get_stub_files())],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=['numpy>=1.16.0'],
    extras_require={
        'mpl': mpl_extras,
        'graphviz': graphviz_extras,
        'all': mpl_extras + graphviz_extras,
    }
)
