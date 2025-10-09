# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import os
import tempfile
import unittest

import rustworkx


class TestMatrixMarket(unittest.TestCase):
    """Test suite for reading and writing Matrix Market data"""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".mtx")
        os.close(fd)
        os.remove(self.path)

    def test_write_to_string(self):
        """Write COO data to a Matrix Market string."""
        nrows, ncols = 3, 3
        rows = [0, 1]
        cols = [1, 2]
        data = [1.0, 1.0]

        mm_str = rustworkx.write_matrix_market_data(nrows, ncols, rows, cols, data, None)
        self.assertIsInstance(mm_str, str)
        self.assertIn("matrixmarket", mm_str)
        self.assertIn("3 3 2", mm_str)

    def test_write_to_file(self):
        """Write COO data to a Matrix Market file."""
        nrows, ncols = 3, 3
        rows = [0, 1]
        cols = [1, 2]
        data = [1.0, 1.0]

        result = rustworkx.write_matrix_market_data(nrows, ncols, rows, cols, data, self.path)
        self.addCleanup(os.remove, self.path)
        self.assertIsNone(result)

        with open(self.path) as f:
            content = f.read()
        self.assertIn("matrixmarket", content)
        self.assertIn("3 3 2", content)

    def test_read_from_file(self):
        """Read a Matrix Market file into COO data."""
        content = """%%MatrixMarket matrix coordinate real general
                    3 3 2
                    1 2 1.0
                    2 3 1.0
                    """
        with open(self.path, "w") as f:
            f.write(content)

        nrows, ncols, rows, cols, data = rustworkx.read_matrix_market_data(self.path, True)
        self.assertEqual((nrows, ncols), (3, 3))
        self.assertEqual(rows, [0, 1])
        self.assertEqual(cols, [1, 2])
        self.assertEqual(data, [1.0, 1.0])

    def test_read_from_string(self):
        """Read Matrix Market data directly from a string."""
        mm_str = """%%MatrixMarket matrix coordinate real general
                    3 3 2
                    1 2 1.0
                    2 3 1.0
                    """
        nrows, ncols, rows, cols, data = rustworkx.read_matrix_market_data(mm_str, False)
        self.assertEqual((nrows, ncols), (3, 3))
        self.assertEqual(rows, [0, 1])
        self.assertEqual(cols, [1, 2])
        self.assertEqual(data, [1.0, 1.0])

    def test_roundtrip_in_memory(self):
        """Roundtrip: write → read should give same COO data."""
        nrows, ncols = 3, 3
        rows = [0, 1]
        cols = [1, 2]
        data = [1.0, 1.0]

        mm_str = rustworkx.write_matrix_market_data(nrows, ncols, rows, cols, data, None)
        n2, m2, r2, c2, d2 = rustworkx.read_matrix_market_data(mm_str, False)

        self.assertEqual((n2, m2), (nrows, ncols))
        self.assertListEqual(r2, rows)
        self.assertListEqual(c2, cols)
        self.assertListEqual(d2, data)

    def test_roundtrip_via_file(self):
        """Roundtrip: write to file and read back should be consistent."""
        nrows, ncols = 2, 2
        rows = [0, 1]
        cols = [1, 0]
        data = [1.0, 2.0]

        rustworkx.write_matrix_market_data(nrows, ncols, rows, cols, data, self.path)
        n2, m2, r2, c2, d2 = rustworkx.read_matrix_market_data(self.path, True)

        self.assertEqual((n2, m2), (2, 2))
        self.assertSetEqual(set(zip(r2, c2)), {(0, 1), (1, 0)})
        self.assertCountEqual(d2, [1.0, 2.0])

