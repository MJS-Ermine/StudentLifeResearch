# 目標洩漏（Target Leakage）自動檢查報告
本報告自動檢查 features 是否含有與目標（如 gpa、grade、score、target、final）相關欄位。

## 發現疑似目標洩漏欄位：
- gpa all

建議：請勿將上述欄位作為特徵進行模型訓練，否則會造成評估失真。
