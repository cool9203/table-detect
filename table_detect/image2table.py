# coding: utf-8
# TODO: 隱性表格線 跟 顯性表格線 怎麼結合
# TODO: 要確認有些表格線框的歪歪的, 且好像有多框還怎麼樣的

import io
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pdf2image
from cv2.typing import MatLike
from img2table.document.base import Document
from img2table.document.base.rotation import estimate_skew, get_connected_components, get_relevant_angles, rotate_img_with_border
from img2table.ocr.base import OCRInstance
from img2table.tables.image import TableImage
from img2table.tables.objects.extraction import ExtractedTable
from matplotlib import pyplot as plt

_root_path = "/mnt/c/Users/ychsu/Downloads"
_filenames = [
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
    document = Document("")

    img = get_image(src=src)

    # image pre-process
    processed_img = img
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # gray_img = cv2.erode(gray_img, kernel)  # 對黑色膨脹(侵蝕白色)
    # gray_img = cv2.dilate(gray_img, kernel)  # 對黑色侵蝕(白色膨脹)

    # processed_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    if show_processed_image:
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("origin")

        plt.subplot(1, 2, 2)
        plt.imshow(processed_img)
        plt.title("processed")

        plt.show()

    doc = TableImage(img=processed_img, **kwds)

    # Table extraction
    tables = doc.extract_tables(
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=False,
    )
    if not tables:
        print("borderless")
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

            image_bytes.seek(0)
            img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rotated_img, rotated = fix_rotation_image(img=img)
            print(f"rotated: {rotated}")

            tables = run_table_detect(src=rotated_img, ocr=None, **kwds)
            # show_table_in_image(src=rotated_img, tables=tables, **kwds)
            show_table_bbox_in_image(src=rotated_img, tables=tables, **kwds)

            image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break


if __name__ == "__main__":
    main(
        dpi=200,
        show_image=True,
    )
