import unittest

from transceiver.helpers.number_parser import parse_number_expr


class TestParseNumberExpr(unittest.TestCase):
    def test_parse_valid_expressions(self) -> None:
        cases = {
            "200e6": 200e6,
            "5.18e9": 5.18e9,
            "3e8/5.18e9": 3e8 / 5.18e9,
            "2*(3+4)": 14.0,
            "-5": -5.0,
            "+7": 7.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_number_expr(text), expected)

    def test_parse_rejects_invalid(self) -> None:
        cases = [
            "",
            "   ",
            "foo",
            "__import__('os').system('echo nope')",
            "1/0",
            "abs(1)",
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_number_expr(text)


if __name__ == "__main__":
    unittest.main()
