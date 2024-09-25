# coding: utf-8

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import cv2
import numpy as np
import pdf2image
from camelot.parsers import Lattice
from matplotlib import pyplot as plt

import utils

logger = utils.get_logger(logger_name=__name__)

_root_path = "/mnt/c/Users/ychsu/Downloads"
_filenames = [
    "大賈-馬禮遜美國學校 1區2F樑(一次).pdf",
    "港洲-陸軍-584旅F棟B區4FL樑版.pdf",
    "太豪-S016-1F車道樑版.pdf",
]


def run_table_detect(
    src: Union[str, Path, bytes, io.BytesIO],
    width: float = None,
    height: float = None,
    **kwds,
):
    parser = Lattice(**kwds)
    parser.pdf_width = width
    parser.pdf_height = height
    parser.imagename = src
    parser._generate_table_bbox()
    return parser


def show_table_in_image(
    src: Union[str, Path, bytes, io.BytesIO],
    tables: Lattice,
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

    for points in [tables.vertical_segments, tables.horizontal_segments]:
        for p in points:
            cv2.line(
                table_img, (int(p[0]), int(tables.pdf_height - p[1])), (int(p[2]), int(tables.pdf_height - p[3])), (0, 0, 255), 2
            )

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
        with TemporaryDirectory() as tempdir:
            for image in images:
                _path = Path(tempdir, f"temp.{image_format}")
                image.save(fp=str(_path), format=image_format)

                tables = run_table_detect(src=str(_path), ocr=None, width=image.width, height=image.height)
                show_table_in_image(src=str(_path), tables=tables)

                print("-" * 25)
        print("*" * 50)
        # break


if __name__ == "__main__":
    main()
