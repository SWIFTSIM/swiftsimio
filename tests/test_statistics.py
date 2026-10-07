"""Test reading SWIFT statistics files."""

from pathlib import Path

from swiftsimio.statistics import SWIFTStatisticsFile, _sanitize_identifier


def test_sanitize_identifier() -> None:
    """Statistics column names become valid, readable Python identifiers."""
    assert _sanitize_identifier("SFR (total)\n") == "sfr_total"
    assert _sanitize_identifier("Gas.mass") == "gasmass"
    assert _sanitize_identifier("2026 financials") == "_2026_financials"
    assert _sanitize_identifier("class") == "class_"
    assert _sanitize_identifier("#") == "_var"
    assert _sanitize_identifier("$$$") == "_var"


def test_statistics_fields_are_accessible_as_attributes(tmp_path: Path) -> None:
    """Fields containing punctuation are available through normal attributes."""
    statistics_file = tmp_path / "SFR.txt"
    statistics_file.write_text(
        "# (0) Time\n"
        "# Unit = dimensionless\n"
        "# (1) Star formation rate\n"
        "# Unit = dimensionless\n"
        "# Time  SFR (total)\n"
        "0.0 1.5\n"
    )

    statistics = SWIFTStatisticsFile(statistics_file)

    assert statistics.header_snake_case_names == ["time", "sfr_total"]
    assert statistics.sfr_total[0] == 1.5
