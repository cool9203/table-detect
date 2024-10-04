# coding: utf-8

import io
from typing import List

from img2table.document import Image as ImageDoc
from img2table.ocr import EasyOCR
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from PIL import Image

from table_detect.base import InputType, get_image, main

ocr = EasyOCR(lang=["ch_tra", "en"])
# ocr = None


def run_table_detect(
    src: InputType,
    ocr: OCRInstance = None,
    implicit_rows=False,
    implicit_columns=False,
    min_confidence=0,
    show_processed_image: bool = False,
    **kwds,
) -> List[ExtractedTable]:
    img = get_image(src=src)
    image_bytes = io.BytesIO()
    Image.fromarray(img).save(fp=image_bytes, format="png")
    document = ImageDoc(image_bytes)

    tables = document.extract_tables(
        ocr=ocr,
        implicit_rows=implicit_rows,
        # implicit_columns=implicit_columns,
        min_confidence=min_confidence,
        borderless_tables=False,
    )

    if not tables:
        tables = document.extract_tables(
            ocr=ocr,
            implicit_rows=implicit_rows,
            # implicit_columns=implicit_columns,
            min_confidence=min_confidence,
            borderless_tables=True,
        )

    return tables


if __name__ == "__main__":
    main(
        dpi=400,
        output_path="./data/result-temp",
        show_image=False,
        show_image_bbox=False,
        save_table_image=True,
        save_table_bbox_image=False,
        save_df_to_xlsx=True,
        run_table_detect=run_table_detect,
    )
