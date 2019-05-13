from setuptools import setup
from setuptools_rust import Binding, RustExtension


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name="retworkx",
    version="0.0.3",
    description="A python graph library implemented in Rust",
    long_description=readme(),
    author = "Matthew Treinish",
    author_email = "mtreinish@kortar.org",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    url="https://github.com/mtreinish/retworkx",
    project_urls={
        "Bug Tracker": "https://github.com/mtreinish/retworkx/issues",
        "Source Code": "https://github.com/mtreinish/retworkx",
        "Documentation": "https://retworkx.readthedocs.io",
    },
    rust_extensions=[RustExtension("retworkx.retworkx", "Cargo.toml",
                                   binding=Binding.PyO3)],
    include_package_data=True,
    packages=["retworkx"],
    zip_safe=False,
    python_requires=">=3.5",
)
