import unittest

from evaluation.recurrent_metrics import exact_match, parse_recurrent_answer


class RecurrentAnswerParserTest(unittest.TestCase):
    def test_parses_answer_tag(self):
        self.assertEqual(
            parse_recurrent_answer("prefix <answer>concept:atdate</answer> suffix"),
            "concept:atdate",
        )

    def test_parses_bracketed_relation(self):
        self.assertEqual(
            parse_recurrent_answer("[concept:atdate]"),
            "concept:atdate",
        )

    def test_canonicalizes_spaced_bracketed_relation(self):
        parsed = parse_recurrent_answer("[concept: bodypart contains bodypart]")
        self.assertEqual(parsed, "concept:bodypartcontainsbodypart")
        self.assertTrue(exact_match(parsed, ["concept:bodypartcontainsbodypart"]))

    def test_answer_tag_takes_priority_over_other_brackets(self):
        self.assertEqual(
            parse_recurrent_answer(
                "[concept:company_pbs] <answer>concept:subpartof</answer>"
            ),
            "concept:subpartof",
        )


if __name__ == "__main__":
    unittest.main()
