from __future__ import annotations

from pathlib import Path

from src.models.ranker import cleanup_processed_intermediate_dirs


def test_cleanup_processed_intermediate_dirs_removes_only_intermediate_dirs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "base_buckets").mkdir()
    (processed_dir / "history_months").mkdir()
    (processed_dir / "training_table.parquet").write_text("keep", encoding="utf-8")

    removed = cleanup_processed_intermediate_dirs(
        {
            "data": {
                "processed_dir": str(processed_dir),
                "training_table": str(processed_dir / "training_table.parquet"),
            }
        }
    )

    assert {path.name for path in removed} == {"base_buckets", "history_months"}
    assert not (processed_dir / "base_buckets").exists()
    assert not (processed_dir / "history_months").exists()
    assert (processed_dir / "training_table.parquet").exists()
