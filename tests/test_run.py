import io
import unittest
from contextlib import redirect_stdout

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


if __name__ == "__main__":
    unittest.main()
