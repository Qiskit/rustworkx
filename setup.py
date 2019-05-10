from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="retworkx",
    version="0.0.1",
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
    rust_extensions=[RustExtension("retworkx.retworkx", "Cargo.toml",
                                   binding=Binding.PyO3)],
    packages=["retworkx"],
    zip_safe=False,
    python_requires=">=3.5",
)
