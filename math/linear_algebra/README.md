# Holberton Linear Algebra & Array Tasks

## Description
This repository contains Python exercises from Holberton School for practicing **linear algebra operations** and **array/list manipulation**. Each task demonstrates a different concept, including matrix shape calculation, matrix transposition, element-wise addition, and list slicing.

---

## Files and Tasks

| File | Task | Function | Description | Example |
|------|------|----------|-------------|---------|
| `0-slice_me_up.py` | List slicing | N/A | Extracts portions of a predefined list: first two numbers, last five numbers, 2nd through 6th numbers. | ```text The first two numbers: [9, 8] The last five numbers: [9, 4, 1, 0, 3] The 2nd through 6th numbers: [8, 2, 3, 9, 4] ``` |
| `2-size_me_please.py` | Matrix shape | `matrix_shape(matrix)` | Calculates the shape of a nested list (matrix) and returns it as a list of integers. | ```python matrix_shape([[1,2],[3,4]]) # [2,2] matrix_shape([[[1,2],[3,4]],[[5,6],[7,8]]]) # [2,2,2] ``` |
| `3-flip_me_over.py` | Matrix transpose | `matrix_transpose(matrix)` | Returns a new 2D matrix that is the transpose of the input. | ```python matrix_transpose([[1,2],[3,4]]) # [[1,3],[2,4]] ``` |
| `4-line_up.py` | Element-wise array addition | `add_arrays(arr1, arr2)` | Returns a new list with sums of two arrays element-wise. Returns `None` if arrays are not the same length. | ```python add_arrays([1,2,3],[4,5,6]) # [5,7,9] add_arrays([1,2],[3,4,5]) # None ``` |

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/holbertonschool-machine_learning.git
