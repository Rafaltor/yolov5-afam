# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:15:13 2022

@author: jamyl and Rafaltor

"""

import torch as th
import random
import cv2

OPENING = +1
CLOSING = -1


class CoverQuery:
    """Segment tree to maintain a set of integer intervals
    and permitting to query the size of their union.
    """

    def __init__(self, L):
        """creates a structure, where all possible intervals
        will be included in [0, L - 1].
        """
        assert L != []  # L is assumed sorted
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)  # --- covered
        self.s = [0] * (2 * self.N)  # --- score
        self.w = [0] * (2 * self.N)  # --- length
        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def cover(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]

    def change(self, i, k, offset):
        """when offset = +1, adds an interval [i, k],
        when offset = -1, removes it
        :complexity: O(log L)
        """
        self._change(1, 0, self.N, i, k, offset)

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:  # --- disjoint
            return
        if i <= start and start + span <= k:  # --- included
            self.c[p] += offset
        else:
            self._change(2 * p, start, span // 2, i, k, offset)
            self._change(2 * p + 1, start + span // 2, span // 2,
                         i, k, offset)
        if self.c[p] == 0:
            if p >= self.N:  # --- leaf
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def get_intersection_from_tuple(rect1, rect2, class_exigence=False):
    """
    Computes the intersection of two rectangles and returns the caracteristics
    of the resulting rectangle

    Parameters
    ----------
    rect1 : Array[5]
        contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
        right corners of the first rectangle
    rect2 : Array[5]
        idem from the second rectangle
    class_exigence : Boolean
        When true, two overlapping boxes will get an intersection surface of 0
        If they have a different class (label)

    Returns
    -------
    intersection : Array[4]
        contains (x1, y1, x2, y2) of the resulting intersection

    """
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])

    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    if x1 <= x2 and y1 <= y2 and not class_exigence:
        intersect = th.tensor([x1, y1, x2, y2], device=rect1.device)
    elif x1 <= x2 and y1 <= y2 and class_exigence and rect1[4] == rect2[4]:
        intersect = th.tensor([x1, y1, x2, y2, rect1[4]], device=rect1.device)
    else:
        intersect = None
    return intersect


def get_intersection_from_list(rectangles, rect2, class_exigence=False):
    """
       Computes the intersection of a list of rectangles and one rectangle and returns the caracteristics list
       of the resulting rectangles

       Parameters
       ----------
       rectangles : list of Array[5]
           contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
           right corners of the first rectangle
       rect2 : Array[5]
           idem from the second rectangle
       class_exigence : Boolean
            When true, two overlapping boxes will get an intersection surface of 0
            If they have a different class (label)

       Returns
       -------
       intersection : list Array[4]
           contains a list of (x1, y1, x2, y2) of the resulting intersection

    """

    inter_boxes = []

    for box in rectangles:

        intersect = get_intersection_from_tuple(
            box, rect2, class_exigence)
        if intersect is not None:
            inter_boxes.append(intersect)
    return inter_boxes


def get_intersection_between_list(rectangles1, rectangles2, class_exigence=False):
    """
       Computes the intersection of a list of rectangles and another list of rectangle and returns the list
       of the resulting rectangles

       Parameters
       ----------
       rectangles1 : list of Array[5]
           contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
           right corners of the first rectangle
       rectangles2 : list of Array[5]
           idem from the second rectangle
       class_exigence : Boolean
            When true, two overlapping boxes will get an intersection surface of 0
            If they have a different class (label)

       Returns
       -------
       intersection : list Array[4]
           contains a list of (x1, y1, x2, y2) of the resulting intersection

    """
    inter_boxes = []

    for box in rectangles2:
        inter_boxes = inter_boxes + (get_intersection_from_list(rectangles1, box, class_exigence))

    return inter_boxes


def get_intersection_area_from_tuple(rect1, rect2, class_exigence=False):
    """
    Returns the area of the intersection between rect1 and rect2

    Parameters
    ----------
    rect1 : Array([5])
        x1, y1, x2, y2, class
    rect2 : Array([5])
        Idem
    class_exigence : Boolean
        When true, two overlapping boxes will get an intersection surface of 0
        If they have a different class (label)

    Returns
    -------
    intersec_area : float
        area of the intersection
    """

    intersect = get_intersection_from_tuple(rect1, rect2, class_exigence)
    intersection_area = box_area(intersect) if intersect is not None else 0
    return intersection_area


def union_area_tuple(rect1, rect2, class_exigence=False):
    return box_area(rect1) + box_area(rect2) - get_intersection_area_from_tuple(rect1, rect2, class_exigence)


def get_union_from_list(R):
    """Area of union of rectangles
    Source: https://github.com/jilljenn/tryalgo.
    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n \\log n)`
    """

    if not R:  # segment tree would fail on an empty list
        return 0
    X = set()  # set of all x coordinates in the input
    events = []  # events for the sweep line
    for Rj in R:
        (x1, y1, x2, y2) = Rj
        assert x1 <= x2 and y1 <= y2
        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))
    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    C = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * C.cover()
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        C.change(i1, i2, offset)
        previous_y = y
    return area
