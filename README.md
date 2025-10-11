[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lyfclldM)
# Homework 2 — AR Cube (3D Computer Vision)

簡短說明
---
此專案為課程作業（Homework 2），實作內容為：

- 2D ↔ 3D 特徵匹配
- 使用 P3P 結合 RANSAC 來估計相機姿態
- 根據姿態將 3D 立方體投影到影像上，產生簡單的 AR（擴增實境）效果

資料集
---
Dataset: [Download](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ)

目錄結構（重點檔案）
---
- `2d3dmathcing.py`  : 2D-3D 特徵匹配流程
- `p3p_ransac_1.py`  : P3P + RANSAC 相機姿態估計
- `transform_cube.py`: 根據姿態計算立方體投影
- `AR.py`           : 最終 AR 顯示 / 影片輸出 (會產生 `output/ar_cube.mp4`)
- `requirements_freeze.txt`: 建議用來安裝所需套件

快速開始（Windows / PowerShell）
---
建議在虛擬環境中執行：

```powershell
# 建立並啟用虛擬環境（如果還沒建立）
conda create -n myenv python=3.11
conda activate myenv

# 安裝相依套件
pip install -r requirements_freeze.txt
```

執行各腳本（示例）
---
請先把 Dataset 解壓或放到 `data/` 資料夾下（如果資料檔名不同，請調整腳本內路徑）。

1) 2D-3D 特徵匹配

```powershell
python 2d3dmathcing.py
```

2) P3P + RANSAC 姿態估計

```powershell
python p3p_ransac_1.py
```

3) 計算立方體投影

```powershell
python transform_cube.py
```

4) 最終 AR 顯示 / 輸出影片

```powershell
python AR.py
# 輸出檔案示例: output/ar_cube.mp4
```

注意事項
---
- repository 中已包含部分輸出 (`output/`) 與資料 (`data/`)。為避免把大型資料上傳到 GitHub，請確認 `.gitignore` 設定已排除不必要的二進位大檔（例如 `.pkl`, `*.npy`, `output/` 等）。
- 若要上傳大型檔案到遠端，請使用 Git LFS（Git Large File Storage）。
- 若執行時遇到路徑錯誤或缺少檔案，請確認 `data/` 裡的檔案位置與程式中使用的路徑一致。

聯絡 / 備註
---
若需要我幫你把這份 README 寫入儲存庫（或調整內容、加上範例參數），請告訴我要寫入的額外細節（例如你想提供的參數、輸入影像範例路徑或想強調的實驗結果）。

---
最後更新：2025-10-11
