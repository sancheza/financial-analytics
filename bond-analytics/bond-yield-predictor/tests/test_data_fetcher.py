#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the data fetcher module.
"""

import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import json
import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher, FRED_SERIES_MAP


class TestDataFetcher(unittest.TestCase):
    """Tests for the DataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test cache directory
        self.test_cache_dir = os.path.join(os.path.dirname(__file__), 'test_cache')
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # Initialize data fetcher with test cache directory
        self.data_fetcher = DataFetcher(cache_dir=self.test_cache_dir)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test cache directory
        for file in os.listdir(self.test_cache_dir):
            os.remove(os.path.join(self.test_cache_dir, file))
        os.rmdir(self.test_cache_dir)

    @patch('src.data_fetcher.requests.get')
    def test_get_fred_data(self, mock_get):
        """Test fetching data from FRED API."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'observations': [
                {'date': '2023-01-01', 'value': '3.50'},
                {'date': '2023-01-02', 'value': '3.55'},
                {'date': '2023-01-03', 'value': '3.60'}
            ]
        }
        mock_get.return_value = mock_response
        
        # Test private _get_fred_data method
        df = self.data_fetcher._get_fred_data('DGS10')
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.columns.tolist(), ['date', 'value'])
        self.assertEqual(df['value'].dtype, float)

    def test_validate_bond_type(self):
        """Test bond type validation."""
        # Test valid bond types
        for bond_type in FRED_SERIES_MAP.keys():
            with self.subTest(bond_type=bond_type):
                self.assertIn(bond_type, FRED_SERIES_MAP)
        
        # Test invalid bond type
        self.assertNotIn('INVALID', FRED_SERIES_MAP)

    @patch('src.data_fetcher.DataFetcher._get_fred_data')
    def test_get_treasury_data_cache(self, mock_get_fred):
        """Test caching functionality."""
        # Create fake data
        fake_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'value': [3.5 + 0.01 * i for i in range(10)]
        })
        mock_get_fred.return_value = fake_data
        
        # First call should fetch from FRED
        df1 = self.data_fetcher.get_treasury_data('10Y', force_update=False)
        mock_get_fred.assert_called_once()
        
        # Second call should use cache
        mock_get_fred.reset_mock()
        df2 = self.data_fetcher.get_treasury_data('10Y', force_update=False)
        mock_get_fred.assert_not_called()
        
        # Force update should fetch from FRED again
        mock_get_fred.reset_mock()
        df3 = self.data_fetcher.get_treasury_data('10Y', force_update=True)
        mock_get_fred.assert_called_once()

    @patch('src.data_fetcher.requests.get')
    def test_get_treasury_direct_auctions(self, mock_get):
        """Test fetching auction data from TreasuryDirect."""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <table>
                <tr>
                    <td>10-year</td>
                </tr>
                <tr>
                    <td>2023-01-15</td>
                    <td>912810SV1</td>
                    <td>3.60</td>
                </tr>
            </table>
        </html>
        """
        mock_get.return_value = mock_response
        
        # Test
        auctions = self.data_fetcher.get_treasury_direct_auctions('10Y', force_update=True)
        
        # Assertions
        self.assertIsInstance(auctions, list)
        self.assertEqual(len(auctions), 1)
        if auctions:
            self.assertIn('date', auctions[0])
            self.assertIn('cusip', auctions[0])
            self.assertIn('high_yield', auctions[0])


if __name__ == '__main__':
    unittest.main()