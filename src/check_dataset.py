import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from src.config import DATA_ROOT

REQUIRED_FILES = [
    DATA_ROOT / "education" / "grades.csv",
    DATA_ROOT / "survey" / "PHQ-9.csv",
    DATA_ROOT / "survey" / "PerceivedStressScale.csv",
    DATA_ROOT / "survey" / "panas.csv",
    DATA_ROOT / "survey" / "psqi.csv",
    DATA_ROOT / "survey" / "vr_12.csv",
    DATA_ROOT / "survey" / "BigFive.csv",
    DATA_ROOT / "survey" / "FlourishingScale.csv",
    DATA_ROOT / "survey" / "LonelinessScale.csv",
]
REQUIRED_DIRS = [
    DATA_ROOT / "sensing" / d for d in [
        "sleep", "phonelock", "activity", "audio", "conversation", "bluetooth", "wifi", "phonecharge", "dark", "app_usage"
    ]
] + [
    DATA_ROOT / "EMA" / "response" / d for d in [
        "Mood", "Activity", "Stress", "Social", "PAM"
    ]
]

def check_dataset():
    print(f"[INFO] 檢查資料集根目錄: {DATA_ROOT}")
    if not DATA_ROOT.exists():
        print(f"[ERROR] 資料集根目錄不存在: {DATA_ROOT}")
        return
    missing = False
    for f in REQUIRED_FILES:
        if not f.exists():
            print(f"[MISSING] 檔案不存在: {f}")
            missing = True
        else:
            print(f"[OK] 檔案存在: {f}")
    for d in REQUIRED_DIRS:
        if not d.exists():
            print(f"[MISSING] 資料夾不存在: {d}")
            missing = True
        else:
            print(f"[OK] 資料夾存在: {d}")
    if not missing:
        print("[SUCCESS] 所有關鍵檔案與資料夾皆存在！")
    else:
        print("[WARNING] 請補齊缺漏的檔案或資料夾後再執行分析！")

if __name__ == "__main__":
    check_dataset() 