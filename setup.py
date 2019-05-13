from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="retworkx",
    version="0.0.2",
    description="A python graph library implemented in Rust",
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
    },
    rust_extensions=[RustExtension("retworkx.retworkx", "Cargo.toml",
                                   binding=Binding.PyO3)],
    include_package_data=True,
    packages=["retworkx"],
    zip_safe=False,
    python_requires=">=3.5",
)
