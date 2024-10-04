# 表格偵測(base opencv)

- [表格偵測(base opencv)](#表格偵測base-opencv)
  - [安裝](#安裝)
  - [啟動與使用指令](#啟動與使用指令)
  - [問題解決](#問題解決)

## 安裝

`pip install -e . && pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless && pip install opencv-contrib-python`

安裝該 package 與修復 opencv 重複安裝問題

## 啟動與使用指令

目前是使用

`python table_detect --root_path <YOUR DATA ROOT PATH> --filenames <YOUR DATA IMAGE FILENAME> <YOUR DATA DIRECTORY NAME>`

使用 **先找出表格與對應儲存格框線, 再根據各儲存格使用 ocr 預測內容** 流程

## 問題解決

`opencv` or `cv2` 相關問題

解決方法: `pip uninstall -y opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless && pip install opencv-contrib-python`

Reference: https://stackoverflow.com/a/61792677
