# coding: utf-8

import io
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import pdf2image
from img2table.document import Image as ImageDoc
from img2table.ocr import EasyOCR
from img2table.ocr.base import OCRInstance
from img2table.tables.objects.extraction import ExtractedTable
from matplotlib import pyplot as plt
from PIL import Image

from table_detect.image2table import InputType, _filenames, _root_path, fix_rotation_image, get_image

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

    return img


def show_table_bbox_in_image(
    src: InputType,
    tables: List[ExtractedTable],
    **kwds,
):
    img = get_image(src=src)
    frontend_img = np.zeros(img.shape, dtype=np.uint8)
    for _, table in enumerate(tables):
        cv2.rectangle(frontend_img, (table.bbox.x1, table.bbox.y1), (table.bbox.x2, table.bbox.y2), (255, 0, 0), -1)
    img = cv2.addWeighted(img, 0.9, frontend_img, 0.3, 1)

    plt.imshow(img)
    plt.show()
    return img


def main(
    image_format: str = "png",
    output_path: str = "./data",
    dpi: int = 200,
    show_image: bool = False,
    show_image_bbox: bool = False,
    save_table_image: bool = False,
    save_table_bbox_image: bool = False,
    save_df_to_xlsx: bool = False,
    *args,
    **kwds,
) -> None:
    all_time = list()
    for filename in _filenames:
        print(filename)
        save_path = Path(output_path, filename)
        save_path.mkdir(parents=True, exist_ok=True)
        path = Path(_root_path, filename)
        image_name_transform = dict()
        if path.suffix == ".pdf":
            images = pdf2image.convert_from_path(str(path), dpi=dpi)
        elif path.suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            images = [Image.open(str(path))]
        elif path.is_dir():
            images = list()
            for _path in path.iterdir():
                print(_path.name)
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

            if show_image or save_table_image:
                _img = show_table_in_image(src=rotated_img, tables=tables, **kwds)
                if show_image:
                    plt.imshow(_img)
                    plt.show()
                if save_table_image:
                    plt.imsave(str(save_path / f"{save_filename}.png"), _img)

            if show_image_bbox or save_table_bbox_image:
                _img = show_table_bbox_in_image(src=rotated_img, tables=tables, **kwds)
                if show_image_bbox:
                    plt.imshow(_img)
                    plt.show()
                if save_table_bbox_image:
                    plt.imsave(str(save_path / f"{save_filename}.png"), _img)

            origin_image_bytes.close()
            print("-" * 25)
        print("*" * 50)
        # break
    print(f"average time: {sum(all_time) / len(all_time)}")


if __name__ == "__main__":
    main(
        dpi=400,
        output_path="./data/result-temp",
        show_image=False,
        show_image_bbox=False,
        save_table_image=True,
        save_table_bbox_image=False,
        save_df_to_xlsx=True,
    )
