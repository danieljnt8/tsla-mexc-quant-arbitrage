"""
app/helpers/report_writer.py
============================
Generates backtest report artifacts in the app/reports/ directory.

Wraps the existing src/reporting/report.py generate_report() function —
no logic duplication.  Only difference: output directory is app/reports/
instead of quant_arbitrage/reports/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR      = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _APP_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.reporting.report import generate_report  # noqa: E402
from src.backtest.engine import EngineResult       # noqa: E402
from config import Config                          # noqa: E402


# Default reports directory inside app/
APP_REPORTS_DIR = _APP_DIR / "reports"


def generate_app_report(
    result: EngineResult,
    cfg: Config,
    start: str,
    end: str,
) -> dict[str, Path]:
    """
    Generate backtest report files in app/reports/.

    Uses the existing generate_report() from src/reporting/report.py.
    Files are named app_backtest_YYYYMMDD_HHMMSS.{ext}.

    Returns
    -------
    dict with keys "md", "png", "csv" pointing to the generated files.
    """
    APP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Override the config's report_dir so files land in app/reports/
    original_report_dir = cfg.report_dir
    cfg.report_dir = str(APP_REPORTS_DIR)

    try:
        md_path = generate_report(
            result=result,
            cfg=cfg,
            start=start,
            end=end,
            report_dir=str(APP_REPORTS_DIR),
        )
    finally:
        cfg.report_dir = original_report_dir

    # generate_report() created a timestamped subfolder; infer its path
    # md_path is like: app/reports/YYYYMMDD_HHMMSS/backtest_YYYYMMDD_HHMMSS.md
    run_dir = md_path.parent
    stem    = md_path.stem.replace("backtest_", "")   # YYYYMMDD_HHMMSS
    png_path = run_dir / f"equity_curve_{stem}.png"
    csv_path = run_dir / f"trades_{stem}.csv"
    pdf_path = run_dir / f"backtest_{stem}.pdf"

    return {
        "md":      md_path,
        "png":     png_path if png_path.exists() else None,
        "csv":     csv_path if csv_path.exists() else None,
        "pdf":     pdf_path if pdf_path.exists() else None,
        "run_dir": run_dir,
    }


def list_app_reports() -> list[dict]:
    """
    Return metadata for all reports saved in app/reports/.

    Each backtest run lives in its own subfolder (app/reports/YYYYMMDD_HHMMSS/).
    Sorted newest first.
    """
    if not APP_REPORTS_DIR.exists():
        return []

    reports = []
    for run_dir in sorted(APP_REPORTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        stem    = run_dir.name   # YYYYMMDD_HHMMSS
        md_file = run_dir / f"backtest_{stem}.md"
        if not md_file.exists():
            continue
        png = run_dir / f"equity_curve_{stem}.png"
        csv = run_dir / f"trades_{stem}.csv"
        pdf = run_dir / f"backtest_{stem}.pdf"
        reports.append({
            "md":        md_file,
            "png":       png if png.exists() else None,
            "csv":       csv if csv.exists() else None,
            "pdf":       pdf if pdf.exists() else None,
            "run_dir":   run_dir,
            "timestamp": stem,
        })
    return reports
