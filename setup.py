from setuptools import setup
try:
    from setuptools_rust import Binding, RustExtension
except ImportError:
    import sys
    import subprocess

    subprocess.call([sys.executable, '-m', 'pip', 'install',
                     'setuptools-rust'])
    from setuptools_rust import Binding, RustExtension


def readme():
    with open('README.md') as f:
        return f.read()


mpl_extras = ['matplotlib>=3.0']
graphviz_extras = ['pydot>=1.4', 'pillow>=5.4']


setup(
    name="retworkx",
    version="0.10.2",
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
    packages=["retworkx", "retworkx.visualization"],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=['numpy>=1.16.0'],
    extras_require={
        'mpl': mpl_extras,
        'graphviz': graphviz_extras,
        'all': mpl_extras + graphviz_extras,
    }
)
