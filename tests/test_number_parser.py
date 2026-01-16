import unittest

from transceiver.helpers.number_parser import parse_number_expr


class ParseNumberExprTests(unittest.TestCase):
    def test_valid_expressions(self) -> None:
        cases = {
            "200e6": 200e6,
            "5.18e9": 5.18e9,
            "3e8/5.18e9": 3e8 / 5.18e9,
            "(1 + 2) * 3": 9.0,
            "-4.5e3": -4500.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_number_expr(text), expected)

    def test_invalid_expressions(self) -> None:
        cases = [
            "",
            " ",
            "os.system('ls')",
            "1 + foo",
            "1/0",
            "().__class__",
            "1e309",
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_number_expr(text)


if __name__ == "__main__":
    unittest.main()
