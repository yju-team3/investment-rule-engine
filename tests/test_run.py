import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from decision_engine import run


class DecisionEngineRunTests(unittest.TestCase):
    def test_run_cli_outputs_sections(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run.main(["--ticker", "PG"])
        output = buffer.getvalue()
        self.assertIn("(1) Decision", output)
        self.assertIn("(2) Reason Log", output)
        self.assertIn("(3) Action Plan", output)
        self.assertIn("PG", output)

    def test_live_mode_waits_on_missing_data(self) -> None:
        buffer = io.StringIO()
        with patch("decision_engine.data_sources.yfinance_source.fetch_ohlcv", return_value=None):
            with redirect_stdout(buffer):
                run.main(["--ticker", "PG", "--mode", "live"])
        output = buffer.getvalue()
        self.assertIn("(1) Decision", output)
        self.assertIn("WAIT", output)
        self.assertIn("라이브 데이터", output)


if __name__ == "__main__":
    unittest.main()
