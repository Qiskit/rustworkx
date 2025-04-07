# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from setuptools import setup


import sys
import pathlib
if sys.version_info >= (3, 11):
    import tomllib
else:
    # setuptools actually depends on this, so we can safely import it
    import tomli as tomllib
    
parent_dir = pathlib.Path(__file__).parent.parent

with open(parent_dir/"README.md") as f:
    original_readme = f.read()
with open(parent_dir/"pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
    
retworkx_readme_compat = f"""# retworkx

`retworkx` is the **deprecated** package name for `rustworkx`. `retworkx` has been
retired and will no longer receive updates. We recommend using `rustworkx` instead,
it should be a drop-in replacement. Thanks to all users that have used `retworkx`
since the early days of the project!

{original_readme}
"""

setup(
    name="retworkx",
    version=pyproject["project"]["version"],
    description=pyproject["project"]["description"],
    long_description=retworkx_readme_compat,
    long_description_content_type="text/markdown",
    author=pyproject["project"]["authors"][0]["name"],
    author_email=pyproject["project"]["authors"][0]["email"],
    license=pyproject["project"]["license"],
    classifiers=pyproject["project"]["classifiers"],
    keywords=pyproject["project"]["keywords"],
    project_urls=pyproject["project"]["urls"],
    python_requires=pyproject["project"]["requires-python"],
    install_requires=["rustworkx"],
    extras_require=pyproject["project"]["optional-dependencies"],
)