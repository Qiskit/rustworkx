import unittest
import rustworkx.sparse6 as rx_sparse6


class TestSparse6(unittest.TestCase):
    def test_explicit_header_rejected(self):
        with self.assertRaises(Exception):
            rx_sparse6.read_sparse6_str('>>sparse6<<:')

    def test_header_payload_rejected(self):
        with self.assertRaises(Exception):
            rx_sparse6.read_sparse6_str('>>sparse6<<:Bg')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
