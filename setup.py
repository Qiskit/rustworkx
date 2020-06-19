from setuptools import setup

if __name__ == "__main__":
    try:
        from setuptools_rust import Binding, RustExtension
    except ImportError:
        import sys
        import subprocess
        subprocess.call([sys.executable, '-m', 'pip', 'install',
                         'setuptools-rust'])
        from setuptools_rust import Binding, RustExtension


    setup(
        rust_extensions=[RustExtension("retworkx.retworkx", "Cargo.toml",
                                       binding=Binding.PyO3)],
    )
