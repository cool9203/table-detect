# coding: utf-8
# TODO: 隱性表格線 跟 顯性表格線 怎麼結合
# TODO: 要確認有些表格線框的歪歪的, 且好像有多框還怎麼樣的

import copy
import io
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pdf2image
from cv2.typing import MatLike
from img2table.document.base.rotation import estimate_skew, get_connected_components, get_relevant_angles, rotate_img_with_border
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from img2table.tables.objects.table import Table
from matplotlib import pyplot as plt
from PIL import Image

from table_detect import _table

_root_path = "/mnt/c/Users/ychsu/Downloads"
_filenames = [
    "table_with_bold_text_and_lines.png",
    "大賈-馬禮遜美國學校 1區2F樑(一次).pdf",
    "港洲-陸軍-584旅F棟B區4FL樑版.pdf",
    "太豪-S016-1F車道樑版.pdf",
]
InputType = Union[str, Path, bytes, io.BytesIO, MatLike]
ocr = None


def get_image(
    src: InputType,
) -> MatLike:
    # Instantiation of document, either an image or a PDF
    if isinstance(src, (MatLike, np.ndarray)):
        img = src
    else:
        if isinstance(src, bytes):
            _src = src
        elif isinstance(src, io.BytesIO):
            src.seek(0)
            _src = src.read()
        elif isinstance(src, (str, Path)):
            with io.open(str(src), "rb") as f:
                _src = f.read()
        img = cv2.imdecode(np.frombuffer(_src, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fix_rotation_image(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees
    :param img: image array
    :return: rotated image array and boolean indicating if the image has been rotated
    """
    # Get connected components of the images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cc_centroids, ref_height, thresh = get_connected_components(img=gray)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return img, False

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    skew_angle = estimate_skew(angles=angles, thresh=thresh)

    if abs(skew_angle) > 0:
        print(f"skew_angle: {skew_angle}")
        return rotate_img_with_border(img=img, angle=skew_angle), True

    return img, False


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


def show_table_in_image(
    src: InputType,
    tables: List[ExtractedTable],
    **kwds,
):
    img = get_image(src=src)

    used_boxes = set()
    for table_index, table in enumerate(tables):
        row_index = 0
        for row in table.content.values():
            for cell in row:
                box_element = (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)
                if box_element in used_boxes:
                    continue
                used_boxes.add(box_element)
                cv2.rectangle(
                    img,
                    (cell.bbox.x1, cell.bbox.y1),
                    (cell.bbox.x2, cell.bbox.y2),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"{table_index}-{row_index}",
                    (cell.bbox.x1, cell.bbox.y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                )
                row_index += 1

    plt.imshow(img)
    plt.show()


def show_table_bbox_in_image(
    src: InputType,
    tables: List[ExtractedTable],
    **kwds,
):
    img = get_image(src=src)
    for _, table in enumerate(tables):
        cv2.rectangle(img, (table.bbox.x1, table.bbox.y1), (table.bbox.x2, table.bbox.y2), (255, 0, 0), 2)

    plt.imshow(img)
    plt.show()


def main(
    image_format: str = "png",
    dpi: int = 200,
    show_image: bool = True,
    show_image_bbox: bool = False,
    *args,
    **kwds,
) -> None:
    for filename in _filenames:
        print(filename)
        path = Path(_root_path, filename)
        if path.suffix == ".pdf":
            images = pdf2image.convert_from_path(str(path), dpi=dpi)
        elif path.suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            images = [Image.open(str(path))]
        else:
            raise TypeError("Not support file extension")

        for image in images:
            origin_image_bytes = io.BytesIO()
            image.save(fp=origin_image_bytes, format=image_format)

            img = get_image(origin_image_bytes)
            rotated_img, rotated = fix_rotation_image(img=img)
            print(f"rotated: {rotated}")

            tables = run_table_detect(src=img, ocr=ocr, **kwds)
            print(tables)

            if show_image:
                show_table_in_image(src=rotated_img, tables=tables, **kwds)
            if show_image_bbox:
                show_table_bbox_in_image(src=rotated_img, tables=tables, **kwds)

            origin_image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break


if __name__ == "__main__":
    main(
        dpi=200,
        show_image=False,
        show_image_bbox=True,
    )
