import os
from pathlib import Path

# 컨테이너에서는 /app/data(볼륨), 로컬에서는 저장소 data/.
DATA_DIR = Path(os.environ.get("MACRO_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
PARQUET = DATA_DIR / "parquet"
PARQUET.mkdir(parents=True, exist_ok=True)
