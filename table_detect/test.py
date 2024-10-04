# coding: utf-8

from typing import List

from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from img2table.tables.objects.table import Table

from table_detect import _table
from table_detect.base import InputType, main

ocr = None


def run_table_detect(
    src: InputType,
    ocr: OCRInstance = None,
    implicit_rows=False,
    implicit_columns=False,
    min_confidence=0,
    show_processed_image: bool = False,
    **kwds,
) -> List[ExtractedTable]:
    tables = _table.detect_table(src=src)
    new_tables = list()
    for table in tables:
        if isinstance(table, Table):
            new_tables.append(table.extracted_table)
        elif isinstance(table, ExtractedTable):
            new_tables.append(table)
        else:
            raise TypeError
    return new_tables


if __name__ == "__main__":
    main(
        dpi=200,
        show_image=False,
        show_image_bbox=True,
        run_table_detect=run_table_detect,
    )
