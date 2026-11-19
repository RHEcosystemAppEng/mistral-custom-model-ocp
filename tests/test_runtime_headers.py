"""Fast checks must read only bounded safetensors headers, never weight data."""

import io
import json
import struct
import unittest


from mistral_validation import runtime_check


class RuntimeHeaderTests(unittest.TestCase):
    def test_stops_before_tensor_payload(self):
        header = {"tensor": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}
        encoded = json.dumps(header).encode()
        stream = io.BytesIO(struct.pack("<Q", len(encoded)) + encoded + b"private-weight-data")
        self.assertEqual(runtime_check.read_header(stream), header)
        self.assertEqual(stream.tell(), 8 + len(encoded))

    def test_rejects_oversized_header_before_allocating_or_reading_payload(self):
        stream = io.BytesIO(struct.pack("<Q", 2**40) + b"unread")
        with self.assertRaises(ValueError):
            runtime_check.read_header(stream)
        self.assertEqual(stream.tell(), 8)

    def test_truncated_or_invalid_headers_fail(self):
        for contents in (b"short", struct.pack("<Q", 32) + b"{}",
                         struct.pack("<Q", 2) + b"[]"):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                runtime_check.read_header(io.BytesIO(contents))


if __name__ == "__main__":
    unittest.main()
