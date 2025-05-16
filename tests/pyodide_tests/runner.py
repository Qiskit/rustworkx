# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

# Running this file is the equivalent of running:
# python -m unittest discover .
# At the root of the tests folder. It works both with Pyodide and
# in a normal Python environment.

import pathlib
import sys
import unittest

def main():
    if sys.platform == "emscripten":
        tests_folder = "/tmp/tests/"
    else:
        # Set the path to the code to test
        tests_folder = pathlib.Path(__file__).parent.parent.resolve()

    # Discover tests in the specified folder
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=tests_folder)

    # Run the discovered tests
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    assert result.wasSuccessful()


if __name__ == "__main__" or sys.platform == "emscripten":
    main()
