# 力新案 - 表格偵測(base opencv)

## 使用

目前是使用

`python -m table_detect.test_table_ocr`

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
