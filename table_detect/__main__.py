# coding: utf-8

import argparse
import pprint


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Table detect(base opencv)")

    parser.add_argument("-r", "--root_path", type=str, required=True, help="輸入檔案的根路徑")
    parser.add_argument("-f", "--filenames", type=str, nargs="+", required=True, help="輸入檔案的名稱(可以是資料夾)")

    parser.add_argument("--dpi", type=int, default=200, help="pdf 檔案類型轉圖片時的 dpi")
    parser.add_argument("--image_format", type=str, default="png", help="pdf to image 時的圖片格式")
    parser.add_argument("--ocr", action="store_true", help="是否使用 ocr")
    parser.add_argument("-o", "--output_path", type=str, default="./data/result", help="儲存結果的路徑")

    parser.add_argument("--save_crop_image", action="store_true", help="是否儲存 crop image")
    parser.add_argument("--crop_image_draw_table_line", action="store_true", help="crop image 是否要畫上表格線")
    parser.add_argument("--show_image", action="store_true", help="執行時顯示帶有表格線的圖片")
    parser.add_argument("--show_image_bbox", action="store_true", help="執行時顯示帶有表格'外框'線的圖片")
    parser.add_argument("--save_table_image", action="store_true", help="是否儲存帶有表格線的圖片")
    parser.add_argument("--save_table_bbox_image", action="store_true", help="是否儲存帶有表格'外框'線的圖片")
    parser.add_argument("--save_df_to_xlsx", action="store_true", help="是否儲存表格結果")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parser()
    args = vars(args)
    ocr = None
    if args.pop("ocr", False):
        from img2table.ocr import EasyOCR

        ocr = EasyOCR(lang=["ch_tra", "en"])

    print("Use parameters:")
    print(pprint.pformat(args))
    print("-" * 50)

    from table_detect.base import main
    from table_detect.table_cell_ocr import run_table_detect

    main(
        run_table_detect=run_table_detect,
        ocr=ocr,
        **args,
    )
