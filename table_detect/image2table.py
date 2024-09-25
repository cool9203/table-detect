# coding: utf-8
# TODO: 隱性表格線 跟 顯性表格線 怎麼結合
# TODO: 要確認有些表格線框的歪歪的, 且好像有多框還怎麼樣的

import io
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pdf2image
from img2table.document.base import Document
from img2table.ocr import EasyOCR
from img2table.ocr.base import OCRInstance
from img2table.tables.image import TableImage
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
    src: Union[str, Path, bytes, io.BytesIO],
    ocr: OCRInstance = None,
    implicit_rows=False,
    implicit_columns=False,
    min_confidence=0,
    **kwds,
) -> List[ExtractedTable]:
    document = Document("")

    # Instantiation of document, either an image or a PDF
    if isinstance(src, bytes):
        _src = src
    elif isinstance(src, io.BytesIO):
        src.seek(0)
        _src = src.read()
    elif isinstance(src, (str, Path)):
        with io.open(str(src), "rb") as f:
            _src = f.read()

    img = cv2.imdecode(np.frombuffer(_src, np.uint8), cv2.IMREAD_COLOR)
    doc = TableImage(img=img, **kwds)

    # Table extraction
    tables = doc.extract_tables(
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=False,
    )
    if not tables:
        tables = doc.extract_tables(
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=True,
        )

    return document.get_table_content(
        tables={0: tables},
        ocr=ocr,
        min_confidence=min_confidence,
    ).get(0)


def show_table_in_image(
    src: Union[str, Path, bytes, io.BytesIO],
    tables: List[ExtractedTable],
):
    if isinstance(src, bytes):
        _src = src
    elif isinstance(src, io.BytesIO):
        src.seek(0)
        _src = src.read()
    elif isinstance(src, (str, Path)):
        with io.open(str(src), "rb") as f:
            _src = f.read()
    table_img = cv2.imdecode(np.fromstring(_src, np.uint8), cv2.IMREAD_COLOR)

    for table_index, table in enumerate(tables):
        row_index = 0
        for row in table.content.values():
            for cell in row:
                cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (0, 0, 255), 2)
                cv2.putText(
                    table_img,
                    f"{table_index}-{row_index}",
                    (cell.bbox.x1, cell.bbox.y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                )
                row_index += 1

    plt.imshow(table_img[:, :, ::-1])  # BGR to RGB
    plt.show()


def main(
    image_format: str = "png",
    dpi: int = 200,
    *args,
    **kwds,
) -> None:
    for filename in _filenames:
        print(filename)
        path = Path(_root_path, filename)
        images = pdf2image.convert_from_path(str(path), dpi=dpi)
        for image in images:
            image_bytes = io.BytesIO()
            image.save(fp=image_bytes, format=image_format)

            tables = run_table_detect(src=image_bytes, ocr=None)
            show_table_in_image(src=image_bytes, tables=tables)

            image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break


if __name__ == "__main__":
    main()
