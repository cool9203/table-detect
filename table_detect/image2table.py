# coding: utf-8
# TODO: 隱性表格線 跟 顯性表格線 怎麼結合
# TODO: 要確認有些表格線框的歪歪的, 且好像有多框還怎麼樣的

from io import BytesIO
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pdf2image
from img2table.document import Image as ImageDoc
from img2table.ocr import EasyOCR
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from matplotlib import pyplot as plt

import utils

logger = utils.get_logger(logger_name=__name__)

_root_path = "/mnt/c/Users/ychsu/Downloads"
_filenames = [
    "大賈-馬禮遜美國學校 1區2F樑(一次).pdf",
    "港洲-陸軍-584旅F棟B區4FL樑版.pdf",
    "太豪-S016-1F車道樑版.pdf",
]
ocr = EasyOCR(lang=["ch_tra", "en"])


def run_table_detect(
    src: Union[str, Path, bytes, BytesIO],
    ocr: OCRInstance = None,
    implicit_rows=False,
    implicit_columns=False,
    min_confidence=0,
) -> List[ExtractedTable]:
    # Instantiation of document, either an image or a PDF
    doc = ImageDoc(src)

    # Table extraction
    extracted_tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        min_confidence=min_confidence,
        borderless_tables=False,
    )
    if not extracted_tables:
        extracted_tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            min_confidence=min_confidence,
            borderless_tables=True,
        )
    return extracted_tables


def show_table_in_image(image_bytes: bytes, tables: List[ExtractedTable]):
    table_img = cv2.imdecode(np.fromstring(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    for table in tables:
        for row in table.content.values():
            for cell in row:
                cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (0, 0, 255), 2)

    plt.imshow(table_img[:, :, ::-1])  # BGR to RGB
    plt.show()


def main(
    image_format: str = "png",
    *args,
    **kwds,
) -> None:
    for filename in _filenames:
        print(filename)
        path = Path(_root_path, filename)
        images = pdf2image.convert_from_path(str(path))
        for image in images:
            image_bytes = BytesIO()
            image.save(fp=image_bytes, format=image_format)
            tables = run_table_detect(src=image_bytes, ocr=None)

            image_bytes.seek(0)
            show_table_in_image(image_bytes=image_bytes.read(), tables=tables)

            image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break


if __name__ == "__main__":
    main()
