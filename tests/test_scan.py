import csv
import os
import tempfile
import unittest
from contextlib import contextmanager

from decision_engine import scan


@contextmanager
def chdir(path: str):
    current = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


class DecisionEngineScanTests(unittest.TestCase):
    def test_scan_creates_csv_with_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with chdir(tmp_dir):
                scan.main(["--mode", "sample", "--tickers", "PG,TSLA,MSFT"])
                results_dir = os.path.join(tmp_dir, "results")
                csv_files = [
                    filename
                    for filename in os.listdir(results_dir)
                    if filename.startswith("scan_") and filename.endswith(".csv")
                ]
                self.assertEqual(len(csv_files), 1)
                csv_path = os.path.join(results_dir, csv_files[0])
                with open(csv_path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
