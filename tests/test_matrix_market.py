# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import os
import tempfile
import unittest

import rustworkx
import numpy as np


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

    def test_dense_and_sparse_consistency(self):
        """Validate consistency between dense and sparse matrices via Matrix Market format."""

        # Dense matrix
        dense = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Convert to COO triplets manually
        nrows, ncols = dense.shape
        rows, cols, vals = [], [], []
        for i in range(nrows):
            for j in range(ncols):
                rows.append(i)
                cols.append(j)
                vals.append(dense[i, j])

        # Write dense as COO -> Matrix Market string
        mm_str = rustworkx.write_matrix_market_data(nrows, ncols, rows, cols, vals, None)

        n2, m2, r2, c2, d2 = rustworkx.read_matrix_market_data(mm_str, False)

        # Assert matrix dimensions
        self.assertEqual((n2, m2), (nrows, ncols))

        # Reconstruct dense from COO
        reconstructed = np.zeros((n2, m2))
        for i, j, v in zip(r2, c2, d2):
            reconstructed[i, j] = v

        # Ensure consistency
        np.testing.assert_allclose(dense, reconstructed)

        # Now test sparse version (only non-zero entries)
        sparse_rows = [0, 2]
        sparse_cols = [1, 2]
        sparse_vals = [10.0, 20.0]
        mm_sparse_str = rustworkx.write_matrix_market_data(
            3, 3, sparse_rows, sparse_cols, sparse_vals, None
        )

        n3, m3, r3, c3, v3 = rustworkx.read_matrix_market_data(mm_sparse_str, False)
        self.assertEqual((n3, m3), (3, 3))
        self.assertEqual(
            sorted(list(zip(r3, c3, v3))), sorted(list(zip(sparse_rows, sparse_cols, sparse_vals)))
        )
