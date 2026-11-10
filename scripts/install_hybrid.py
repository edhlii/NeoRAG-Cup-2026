"""Cài pipeline hybrid bằng PyTorch CPU; không tải CUDA hoặc toàn bộ repo model."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Hãy kích hoạt .venv hoặc môi trường conda trước khi cài.")
    for args in (
        ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
        ["-e", ".[app,dev,hybrid]"],
    ):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", *args],
            cwd=ROOT,
            check=True,
        )
    if "--download-models" in sys.argv:
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
        from neorag.hybrid.models import download_model
        from neorag.hybrid.settings import Settings

        cfg = Settings.load()
        for kind in ("embedding", "reranker"):
            download_model(cfg.models[kind], cfg.models.get(f"{kind}_revision", "main"))
    print("Sẵn sàng. Chạy: docker compose up -d && xem docs/hybrid.md")


if __name__ == "__main__":
    main()
