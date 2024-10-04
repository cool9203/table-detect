# coding: utf-8

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from cv2.typing import MatLike
from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.bordered_tables.cells import get_cells
from img2table.tables.processing.bordered_tables.tables.cell_clustering import cluster_cells_in_tables
from img2table.tables.processing.bordered_tables.tables.semi_bordered import add_semi_bordered_cells
from img2table.tables.processing.bordered_tables.tables.table_creation import cluster_to_table, normalize_table_cells
from matplotlib import pyplot as plt

from table_detect.metrics import compute_img_metrics

InputType = Union[str, Path, bytes, io.BytesIO, MatLike]


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


def processing_image(img: np.ndarray) -> MatLike:
    return img


def identify_straight_lines(
    thresh: np.ndarray,
    min_line_length: float,
    char_length: float,
    vertical: bool = True,
) -> List[Line]:
    """
    Identify straight lines in image in a specific direction
    :param thresh: thresholded edge image
    :param min_line_length: minimum line length
    :param char_length: average character length
    :param vertical: boolean indicating if vertical lines are detected
    :return: list of detected lines
    """
    # Apply masking on image
    kernel_dims = (1, round(min_line_length / 3)) if vertical else (round(min_line_length / 3), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply closing for hollow lines
    hollow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if vertical else (1, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hollow_kernel)

    # Apply closing for dotted lines
    dotted_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, round(min_line_length / 6)) if vertical else (round(min_line_length / 6), 1)
    )
    mask_dotted = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, dotted_kernel)

    # Apply masking on line length
    kernel_dims = (1, min_line_length) if vertical else (min_line_length, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    final_mask = cv2.morphologyEx(mask_dotted, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get stats
    _, _, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8, cv2.CV_32S)

    lines = list()
    # Get relevant CC that correspond to lines
    for idx, stat in enumerate(stats):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = stat

        # Filter on aspect ratio
        if max(w, h) / min(w, h) < 5 and min(w, h) >= char_length:
            continue
        # Filter on length
        if max(w, h) < min_line_length:
            continue

        cropped = thresh[y : y + h, x : x + w]
        if w >= h:
            non_blank_pixels = np.where(np.sum(cropped, axis=0) > 0)
            line_rows = np.where((np.sum(cropped, axis=1) / 255) >= 0.5 * w)

            if len(line_rows[0]) == 0:
                continue

            line = Line(
                x1=x + np.min(non_blank_pixels),
                y1=y + round(np.mean(line_rows)),
                x2=x + np.max(non_blank_pixels),
                y2=y + round(np.mean(line_rows)),
                thickness=np.max(line_rows) - np.min(line_rows) + 1,
            )
        else:
            non_blank_pixels = np.where(np.sum(cropped, axis=1) > 0)
            line_cols = np.where((np.sum(cropped, axis=0) / 255) >= 0.5 * h)

            if len(line_cols[0]) == 0:
                continue

            line = Line(
                x1=x + round(np.mean(line_cols)),
                y1=y + np.min(non_blank_pixels),
                x2=x + round(np.mean(line_cols)),
                y2=y + np.max(non_blank_pixels),
                thickness=np.max(line_cols) - np.min(line_cols) + 1,
            )
        lines.append(line)

    return lines


def identify_straight_lines_custom(
    thresh: np.ndarray,
    line_scale: int = 30,
    vertical: bool = True,
    iterations: int = 1,
    min_line_length: int = None,
    **kwds,
) -> Tuple[MatLike, List[Line]]:
    lines = list()
    min_line_length = (
        min_line_length
        if min_line_length
        else (int(thresh.shape[0] / line_scale) if not vertical else int(thresh.shape[1] / line_scale))
    )
    ksize = (min_line_length, 1) if not vertical else (1, min_line_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations)

    ksize = (3, 1) if not vertical else (1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    detect = cv2.dilate(detect, ksize, iterations=3)  # 對黑色侵蝕(白色膨脹)
    detect = cv2.erode(detect, ksize, iterations=3)  # 對黑色膨脹(侵蝕白色)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(detect, 8, cv2.CV_32S)
    for index in range(1, num_labels):
        x, y, w, h, area = stats[index]
        mask: np.ndarray = labels == index

        if (w >= min_line_length and not vertical) or (h >= min_line_length and vertical):
            lines.append(
                Line(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    thickness=3,
                )
            )
    return (detect, lines)


def threshold_dark_areas(img: np.ndarray, char_length: Optional[float]) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # If image is mainly black, revert the image
    if np.mean(gray) <= 127:
        gray = 255 - gray

    thresh_kernel = int(char_length) // 2 * 2 + 1

    # Threshold original image
    t_sauvola = cv2.ximgproc.niBlackThreshold(
        gray, 255, cv2.THRESH_BINARY_INV, thresh_kernel, 0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA
    )
    thresh = 255 * (gray <= t_sauvola).astype(np.uint8)
    binary_thresh = None

    # Mask on areas with dark background
    blur_size = min(255, int(2 * char_length) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    mask = cv2.inRange(blur, 0, 100)

    # Identify dark areas
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    # For each dark area, use binary threshold instead of regular threshold
    for idx, row in enumerate(stats):
        # Get statistics
        x, y, w, h, area = row

        if idx == 0:
            continue

        if area / (w * h) >= 0.5 and min(w, h) >= char_length and max(w, h) >= 5 * char_length:
            if binary_thresh is None:
                # Threshold binary image
                bin_t_sauvola = cv2.ximgproc.niBlackThreshold(
                    255 - gray,
                    255,
                    cv2.THRESH_BINARY_INV,
                    thresh_kernel,
                    0.2,
                    binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
                )
                binary_thresh = 255 * (255 - gray <= bin_t_sauvola).astype(np.uint8)
            thresh[y : y + h, x : x + w] = binary_thresh[y : y + h, x : x + w]

    return thresh


def detect_text(
    img: np.ndarray,
    char_length: int = 11,
) -> List[Cell]:
    thresh = threshold_dark_areas(img, char_length)
    return compute_img_metrics(thresh=thresh)[2]


def detect_lines(
    img: np.ndarray,
    line_scale: int = 30,
    iterations: int = 1,
    min_line_length: int = 15,
    text_contours: Optional[List[Cell]] = [],
) -> Tuple[List[Line], List[Line]]:
    result = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge - laplacian
    edge = cv2.Laplacian(blur, cv2.CV_64F)

    # Edge - sobel
    # sobelX = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    # sobelY = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    # sobelX = np.uint8(np.absolute(sobelX))
    # sobelY = np.uint8(np.absolute(sobelY))
    # edge = cv2.bitwise_or(sobelX, sobelY)

    edge = np.uint8(np.absolute(edge))
    # thresh = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(edge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Remove text contours
    for c in text_contours:
        thresh[c.y1 - 1 : c.y2 + 1, c.x1 - 1 : c.x2 + 1] = 0

    # Show pre-process image
    plt.subplot(1, 2, 1)
    plt.imshow(edge)
    plt.title("edge")

    plt.subplot(1, 2, 2)
    plt.imshow(thresh)
    plt.title("thresh")

    plt.show()

    (horizontal_line_img, h_lines) = identify_straight_lines_custom(
        thresh=thresh,
        line_scale=line_scale,
        min_line_length=min_line_length,
        iterations=iterations,
        vertical=False,
    )
    (vertical_line_img, v_lines) = identify_straight_lines_custom(
        thresh=thresh,
        line_scale=line_scale,
        min_line_length=min_line_length,
        iterations=iterations,
        vertical=True,
    )
    for line in h_lines + v_lines:
        cv2.rectangle(result, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 2)

    plt.subplot(1, 2, 1)
    plt.imshow(horizontal_line_img)
    plt.title("horizontal_line_img")

    plt.subplot(1, 2, 2)
    plt.imshow(vertical_line_img)
    plt.title("vertical_line_img")

    plt.show()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("origin")

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("result")

    plt.show()

    return (h_lines, v_lines)


def get_tables(cells: List[Cell], elements: List[Cell], lines: List[Line], char_length: float) -> List[Table]:
    """
    Identify and create Table object from list of image cells
    :param cells: list of cells found in image
    :param elements: list of image elements
    :param lines: list of image lines
    :param char_length: average character length
    :return: list of Table objects inferred from cells
    """
    # Cluster cells into tables
    list_cluster_cells = cluster_cells_in_tables(cells=cells)

    # Normalize cells in clusters
    clusters_normalized = [normalize_table_cells(cluster_cells=cluster_cells) for cluster_cells in list_cluster_cells]

    # Add semi-bordered cells to clusters
    complete_clusters = [
        add_semi_bordered_cells(cluster=cluster, lines=lines, char_length=char_length)
        for cluster in clusters_normalized
        if len(cluster) > 0
    ]

    # Create tables from cells clusters
    tables = [cluster_to_table(cluster_cells=cluster, elements=elements) for cluster in complete_clusters]

    return tables


def detect_table(
    src: InputType,
    **kwds,
):
    origin_img = get_image(src=src)

    img = processing_image(img=origin_img)
    text_contours = detect_text(img)
    (h_lines, v_lines) = detect_lines(img=img, text_contours=text_contours)
    # Create cells from rows
    cells = get_cells(horizontal_lines=h_lines, vertical_lines=v_lines)

    # Create tables from rows
    return get_tables(cells=cells, elements=text_contours, lines=v_lines + h_lines, char_length=11)
