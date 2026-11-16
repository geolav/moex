"""Regression tests for issuer-specific RSS matching and risk headline parsing."""
import unittest

from news_filter import classify_headlines, matching_risk_headlines, news_matches_issuer


class NewsFilterTests(unittest.TestCase):
    def test_generic_first_word_does_not_match_vtbr(self):
        self.assertFalse(news_matches_issuer(
            "Банк объявил о дефолте по облигациям", "VTBR", "Банк ВТБ (ПАО)"))

    def test_vtb_alias_matches_the_issuer(self):
        self.assertTrue(news_matches_issuer(
            "ВТБ сообщил о результатах проверки", "VTBR", "Банк ВТБ (ПАО)"))
        self.assertTrue(news_matches_issuer(
            "VTB reports quarterly results", "VTBR", "Банк ВТБ (ПАО)"))

    def test_exact_ticker_and_company_names_match(self):
        self.assertTrue(news_matches_issuer("VTBR: default risk rises", "VTBR", "Банк ВТБ"))
        self.assertTrue(news_matches_issuer("Газпром сообщил о дивидендах", "GAZP", "ПАО Газпром"))
        self.assertTrue(news_matches_issuer("Severstal announces results", "CHMF", "Северсталь"))

    def test_other_bank_default_does_not_become_vtbr_news(self):
        headline = "Региональный банк допустил дефолт по облигациям"
        self.assertFalse(news_matches_issuer(headline, "VTBR", "Банк ВТБ"))

    def test_risk_classifier_returns_the_actual_triggering_headline(self):
        headlines = [
            "ВТБ сообщил о росте чистой прибыли",
            "ВТБ заявил о банкротстве дочерней компании",
        ]
        score, reasons = classify_headlines(headlines)
        self.assertEqual(score, -0.70)
        self.assertIn("дефолт/банкротство", reasons)
        matches = matching_risk_headlines(headlines)
        self.assertEqual(len(matches), 1)
        self.assertIn("банкротстве дочерней компании", matches[0][1])

    def test_separate_headlines_cannot_form_a_split_risk_phrase(self):
        score, reasons = classify_headlines([
            "Компания сообщает о финансовых результатах",
            "Проводится независимое исследование отрасли",
        ])
        self.assertEqual(score, 0.0)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
