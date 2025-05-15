import pathlib


def find_only_pyodide_wheel(directory):
    """
    Find the only pyodide wheel in the current directory.
    Throws an error if there is not exactly one .whl file,
    to remind users to clean up the directory.
    """
    # Filter for .whl files
    whl_files = [
        f for f  in directory.iterdir()
        if f.is_file() and f.suffix == ".whl"
    ]
    pyodide_whl_files = [
        f for f in whl_files
        if "rustworkx" in f.name and "pyodide" in f.name
    ]

    # Check if there is exactly one .whl file
    if len(whl_files) != 1:
        raise RuntimeError(
            "There should be exactly one .whl file in the dist/ directory. Please clean up the directory."
        )

    return whl_files[0]  # Return the name of the .whl file

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__)
    dist_dir = pathlib.Path(__file__).parent / "../dist"
    absolute_dist_path = dist_dir.resolve()

    # Find the only pyodide wheel
    wheel_path = find_only_pyodide_wheel(absolute_dist_path)

    # Print the name of the wheel file
    print(wheel_path)