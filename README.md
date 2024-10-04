# 力新案 - 表格偵測(base opencv)

## 安裝

`pip install -e . && pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless && pip install opencv-contrib-python`

安裝該 package 與修復 opencv 重複安裝問題

## 啟動與使用指令

目前是使用

`python -m table_detect.test_table_ocr`

使用 **先找出表格與對應儲存格框線, 再根據各儲存格使用 ocr 預測內容** 流程

### 資料路徑修改

`table_detect.image2table` 的 `_root_path`, `_filenames` 修改路徑

## 問題解決

img2table error

需要直接到 `img2table.document.base.rotation line 206` 修改成 `height, width, channel = img.shape`

因為 pass 的 img shape 不是灰階圖, 需要是原本的 3 channel 圖

---

`opencv` or `cv2` 相關問題

解決方法: `pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless && pip install opencv-contrib-python`

Reference: https://stackoverflow.com/a/61792677
