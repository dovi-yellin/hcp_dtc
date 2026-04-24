"""Smoke tests for the transitional package/import layout."""

import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_infrastructure_imports(self):
        from infrastructure.filters import butter_bandpass_filter
        from infrastructure.init_params import initParams
        from infrastructure.rate_model import RateModel

        self.assertIsNotNone(butter_bandpass_filter)
        self.assertIsNotNone(initParams)
        self.assertIsNotNone(RateModel)

    def test_legacy_wrappers_import(self):
        from scripts.initParams import initParams
        from scripts.models import RateModel
        from scripts.utils import butter_bandpass_filter

        self.assertIsNotNone(butter_bandpass_filter)
        self.assertIsNotNone(initParams)
        self.assertIsNotNone(RateModel)


if __name__ == "__main__":
    unittest.main()
