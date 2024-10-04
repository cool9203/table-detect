# coding: utf-8

import io
from typing import List

from img2table.document import Image as ImageDoc
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from PIL import Image

from table_detect.base import (
    InputType,
    get_image,
    main,
)

# ocr = EasyOCR(lang=["ch_tra", "en"])
ocr = None


def run_table_detect(
    src: InputType,
    ocr: OCRInstance = None,
    implicit_rows=False,
    min_confidence=0,
    **kwds,
) -> List[ExtractedTable]:
    img = get_image(src=src)
    image_bytes = io.BytesIO()
    Image.fromarray(img).save(fp=image_bytes, format="png")
    document = ImageDoc(image_bytes)

    tables = document.extract_tables(
        ocr=None,
        implicit_rows=implicit_rows,
        # implicit_columns=implicit_columns,
        min_confidence=min_confidence,
        borderless_tables=False,
    )

    if not tables:
        tables = document.extract_tables(
            ocr=None,
            implicit_rows=implicit_rows,
            # implicit_columns=implicit_columns,
            min_confidence=min_confidence,
            borderless_tables=True,
        )

    # Predict table cell text content
    used_boxes = set()
    for table in tables:
        for row in table.content.values():
            for cell in row:
                box_element = (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)
                if box_element in used_boxes or not ocr:
                    continue
                used_boxes.add(box_element)

                if (cell.bbox.y2 - cell.bbox.y1 == 0) or (cell.bbox.x2 - cell.bbox.x1 == 0):
                    continue
                words = ocr.reader.readtext(img[cell.bbox.y1 : cell.bbox.y2, cell.bbox.x1 : cell.bbox.x2])
                for word in words:
                    if round(100 * word[2]) >= min_confidence:
                        cell.value = word[1]
                    else:
                        cell.value = None
                    break

    return tables


if __name__ == "__main__":
    main(
        dpi=400,
        output_path="./data/result-temp",
        show_image=False,
        show_image_bbox=False,
        save_table_image=False,
        save_table_bbox_image=False,
        save_df_to_xlsx=False,
        save_crop_image=True,
        crop_image_draw_table_line=True,
        run_table_detect=run_table_detect,
    )
