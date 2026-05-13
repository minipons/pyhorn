"""Unit tests for pyhorn.output.exporter."""

import csv
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from pyhorn_core.output.exporter import export_to_csv, export_to_json


class TestExportToCsv:
    """Tests for export_to_csv()."""

    @pytest.fixture
    def freqs(self):
        return np.array([100.0, 200.0, 500.0, 1000.0])

    @pytest.fixture
    def spl_responses(self):
        return {
            "total": np.array([85.0, 88.0, 92.0, 95.0]),
            "direct": np.array([80.0, 82.0, 85.0, 87.0]),
        }

    def test_writes_csv_file(self, freqs, spl_responses, tmp_path):
        """Should create the output CSV file."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        assert out.exists()

    def test_csv_has_frequency_header(self, freqs, spl_responses, tmp_path):
        """CSV first column should be labelled 'Frequency_Hz'."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        with open(out) as f:
            header = f.readline().strip()
        assert "Frequency_Hz" in header

    def test_csv_row_count_matches_freqs(self, freqs, spl_responses, tmp_path):
        """CSV should have one data row per frequency entry."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        with open(out) as f:
            rows = list(csv.reader(f))
        assert len(rows) == len(freqs) + 1  # +1 for header

    def test_csv_frequency_values(self, freqs, spl_responses, tmp_path):
        """Frequency column should match input values."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        with open(out) as f:
            rows = list(csv.reader(f))
        for i, freq in enumerate(freqs):
            assert float(rows[i + 1][0]) == pytest.approx(freq, rel=1e-6)

    def test_csv_spl_labels_in_header(self, freqs, spl_responses, tmp_path):
        """SPL response keys should appear as column headers."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        with open(out) as f:
            header = f.readline().strip()
        assert "total" in header
        assert "direct" in header

    def test_empty_response_dict(self, freqs, tmp_path):
        """Empty responses dict should produce a CSV with only frequency column."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, {}, out)
        with open(out) as f:
            rows = list(csv.reader(f))
        assert len(rows) == len(freqs) + 1
        # Only frequency column, no SPL columns
        assert len(rows[0]) == 1

    def test_csv_overwrites_existing(self, freqs, spl_responses, tmp_path):
        """Writing twice to the same path should succeed (overwrite)."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out)
        export_to_csv(freqs, spl_responses, out)
        with open(out) as f:
            rows = list(csv.reader(f))
        assert len(rows) == len(freqs) + 1

    def test_csv_futtrup_gdlimit_in_header(self, freqs, spl_responses, tmp_path):
        """When futtrup_gdlimit_ms is provided, it appears in the CSV header."""
        futtrup_gd = np.array([15.0, 12.0, 8.0, 5.0])
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out, futtrup_gdlimit_ms=futtrup_gd)
        with open(out) as f:
            header = f.readline().strip()
        assert "Futtrup_GDlimit_ms" in header

    def test_csv_futtrup_gdlimit_values_match(self, freqs, spl_responses, tmp_path):
        """futtrup_gdlimit_ms values should appear verbatim in the correct column."""
        futtrup_gd = np.array([15.0, 12.0, 8.0, 5.0])
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out, futtrup_gdlimit_ms=futtrup_gd)
        with open(out) as f:
            rows = list(csv.reader(f))
        header = rows[0]
        gd_col_idx = header.index("Futtrup_GDlimit_ms")
        for i, val in enumerate(futtrup_gd):
            assert float(rows[i + 1][gd_col_idx]) == pytest.approx(val, rel=1e-6)

    def test_csv_futtrup_gdlimit_absent_when_none(self, freqs, spl_responses, tmp_path):
        """When futtrup_gdlimit_ms is None, the column should not appear."""
        out = tmp_path / "out.csv"
        export_to_csv(freqs, spl_responses, out, futtrup_gdlimit_ms=None)
        with open(out) as f:
            header = f.readline().strip()
        assert "Futtrup_GDlimit_ms" not in header


class TestExportToJson:
    """Tests for export_to_json()."""

    @pytest.fixture
    def freqs(self):
        return np.array([100.0, 200.0, 500.0])

    @pytest.fixture
    def spl_responses(self):
        return {
            "total": np.array([85.0, 88.0, 92.0]),
            "direct": np.array([80.0, 82.0, 85.0]),
        }

    def test_writes_json_file(self, freqs, spl_responses, tmp_path):
        """Should create the output JSON file."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        assert out.exists()

    def test_json_contains_frequencies_key(self, freqs, spl_responses, tmp_path):
        """JSON should have a 'frequencies' key."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        assert "frequencies" in data

    def test_json_contains_responses_key(self, freqs, spl_responses, tmp_path):
        """JSON should have a 'responses' key."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        assert "responses" in data

    def test_json_frequencies_are_list(self, freqs, spl_responses, tmp_path):
        """frequencies should be a list (not numpy array)."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data["frequencies"], list)
        assert len(data["frequencies"]) == len(freqs)

    def test_json_responses_have_labels(self, freqs, spl_responses, tmp_path):
        """Each response label should appear as a key in the responses dict."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        assert "total" in data["responses"]
        assert "direct" in data["responses"]

    def test_json_response_arrays_are_lists(self, freqs, spl_responses, tmp_path):
        """Response arrays should be JSON-readable lists."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        for label, arr in data["responses"].items():
            assert isinstance(arr, list)
            assert len(arr) == len(freqs)

    def test_empty_responses_dict(self, freqs, tmp_path):
        """Empty responses should produce valid JSON with empty responses dict."""
        out = tmp_path / "out.json"
        export_to_json(freqs, {}, out)
        with open(out) as f:
            data = json.load(f)
        assert "frequencies" in data
        assert "responses" in data
        assert len(data["responses"]) == 0

    def test_json_overwrites_existing(self, freqs, spl_responses, tmp_path):
        """Writing twice to the same path should succeed (overwrite)."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out)
        export_to_json(freqs, spl_responses, out)
        with open(out) as f:
            data = json.load(f)
        assert "total" in data["responses"]

    def test_json_futtrup_gdlimit_key_present(self, freqs, spl_responses, tmp_path):
        """When futtrup_gdlimit_ms is provided, it appears as a top-level key."""
        futtrup_gd = np.array([15.0, 12.0, 8.0])
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out, futtrup_gdlimit_ms=futtrup_gd)
        with open(out) as f:
            data = json.load(f)
        assert "futtrup_gdlimit_ms" in data
        assert data["futtrup_gdlimit_ms"] == pytest.approx(futtrup_gd.tolist())

    def test_json_futtrup_gdlimit_absent_when_none(self, freqs, spl_responses, tmp_path):
        """When futtrup_gdlimit_ms is None, the key should not appear in JSON."""
        out = tmp_path / "out.json"
        export_to_json(freqs, spl_responses, out, futtrup_gdlimit_ms=None)
        with open(out) as f:
            data = json.load(f)
        assert "futtrup_gdlimit_ms" not in data