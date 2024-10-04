# coding: utf-8

import io
import time
import warnings
from pathlib import Path
from typing import Callable, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pdf2image
from cv2.typing import MatLike
from img2table.document.base.rotation import estimate_skew, get_connected_components, get_relevant_angles
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from matplotlib import pyplot as plt
from PIL import Image

InputType = Union[str, Path, bytes, io.BytesIO, MatLike]
_root_path = "/mnt/c/Users/ychsu/Downloads"
_filenames = [
    "AIOCR",
    "table-pic",
    "大賈-馬禮遜美國學校 1區2F樑(一次).pdf",
    "港洲-陸軍-584旅F棟B區4FL樑版.pdf",
    "太豪-S016-1F車道樑版.pdf",
]
warnings.simplefilter("ignore")


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


def rotate_img_with_border(
    img: np.ndarray,
    angle: float,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Rotate an image of the defined angle and add background on border
    :param img: image array
    :param angle: rotation angle
    :param background_color: background color for borders after rotation
    :return: rotated image array
    """
    # Compute image center
    height, width = (img.shape[0], img.shape[1])
    image_center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Get rotated image dimension
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))

    # Update rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Create rotated image with white background
    rotated_img = cv2.warpAffine(
        img, rotation_mat, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=background_color
    )
    return rotated_img


def fix_rotation_image(
    img: np.ndarray,
) -> Tuple[np.ndarray, bool]:
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


def draw_table_in_image(
    src: InputType,
    tables: List[ExtractedTable],
    show_cell_text: bool = True,
    **kwds,
) -> MatLike:
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
                if show_cell_text:
                    cv2.putText(
                        img,
                        f"{table_index}-{row_index}",
                        (cell.bbox.x1, cell.bbox.y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                    )
                row_index += 1

    return img


def crop_table_bbox(
    src: InputType,
    tables: List[ExtractedTable],
    **kwds,
) -> List[MatLike]:
    img = get_image(src=src)
    crop_img = list()
    for table in tables:
        crop_img.append(img[table.bbox.y1 : table.bbox.y2, table.bbox.x1 : table.bbox.x2])

    return crop_img


def draw_table_bbox_in_image(
    src: InputType,
    tables: List[ExtractedTable],
    **kwds,
) -> MatLike:
    img = get_image(src=src)
    frontend_img = np.zeros(img.shape, dtype=np.uint8)
    for _, table in enumerate(tables):
        cv2.rectangle(frontend_img, (table.bbox.x1, table.bbox.y1), (table.bbox.x2, table.bbox.y2), (255, 0, 0), -1)
    img = cv2.addWeighted(img, 0.9, frontend_img, 0.3, 1)

    return img


def main(
    root_path: Union[str, Path] = _root_path,
    filenames: List[Union[str, Path]] = _filenames,
    image_format: str = "png",
    output_path: str = "./data",
    dpi: int = 200,
    save_crop_image: bool = False,
    crop_image_draw_table_line: bool = False,
    show_image: bool = False,
    show_image_bbox: bool = False,
    save_table_image: bool = False,
    save_table_bbox_image: bool = False,
    save_df_to_xlsx: bool = False,
    ocr: OCRInstance = None,
    run_table_detect: Callable[..., List[ExtractedTable]] = None,
    *args,
    **kwds,
) -> None:
    if not run_table_detect:
        raise RuntimeError("run_table_detect function must be pass")

    root_path = Path(root_path)
    output_path = Path(output_path)

    if not root_path.exists():
        raise NotADirectoryError("root_path not a directory")

    for filename in filenames:
        if not Path(root_path, filename).exists():
            raise FileNotFoundError(f"'{filename}' not found")

    all_time = list()
    for filename in filenames:
        print(filename)
        save_path = Path(output_path, filename)
        save_path.mkdir(parents=True, exist_ok=True)
        path = Path(root_path, filename)

        image_name_transform = dict()
        if path.suffix == ".pdf":
            images = pdf2image.convert_from_path(str(path), dpi=dpi)
        elif path.suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            images = [Image.open(str(path))]
        elif path.is_dir():
            images = list()
            for _path in path.iterdir():
                if _path.suffix == ".pdf":
                    images += pdf2image.convert_from_path(str(_path), dpi=dpi)
                elif _path.suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
                    image_name_transform[len(images)] = _path.stem
                    images.append(Image.open(str(_path)))
                else:
                    raise TypeError("Not support file extension")
        else:
            raise TypeError("Not support file extension")

        for index, image in enumerate(images):
            if index in image_name_transform:
                print(image_name_transform.get(index))

            origin_image_bytes = io.BytesIO()
            image.save(fp=origin_image_bytes, format=image_format)

            img = get_image(origin_image_bytes)
            rotated_img, rotated = fix_rotation_image(img=img)
            print(f"rotated: {rotated}")

            start_time = time.time()
            tables = run_table_detect(src=img, ocr=ocr, **kwds)
            all_time.append(time.time() - start_time)
            print(f"time: {all_time[-1]}")

            save_filename = image_name_transform.get(index, index)
            if save_df_to_xlsx:
                writer = pd.ExcelWriter(str(save_path / f"{save_filename}.xlsx"))
                for table_index, table in enumerate(tables):
                    table.df.to_excel(writer, sheet_name=str(table_index), index=False, header=False)
                writer.close()

            if save_crop_image:
                if crop_image_draw_table_line:
                    _img = draw_table_in_image(src=rotated_img, tables=tables, show_cell_text=False, **kwds)
                else:
                    _img = rotated_img
                crop_table_images = crop_table_bbox(src=_img, tables=tables, **kwds)
                for i, crop_table_image in enumerate(crop_table_images):
                    plt.imsave(str(save_path / f"{save_filename}-crop-table-{i}.png"), crop_table_image)

            if show_image or save_table_image:
                _img = draw_table_in_image(src=rotated_img, tables=tables, **kwds)
                if show_image:
                    plt.imshow(_img)
                    plt.show()
                if save_table_image:
                    plt.imsave(str(save_path / f"{save_filename}-table.png"), _img)

            if show_image_bbox or save_table_bbox_image:
                _img = draw_table_bbox_in_image(src=rotated_img, tables=tables, **kwds)
                if show_image_bbox:
                    plt.imshow(_img)
                    plt.show()
                if save_table_bbox_image:
                    plt.imsave(str(save_path / f"{save_filename}-bbox.png"), _img)

            origin_image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break
    print(f"average time: {sum(all_time) / len(all_time)}")
