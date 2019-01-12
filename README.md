# MorphOpsCUDA
Mathematical morphological operators implementation in CUDA (parallel version), three versions are avaiable: check out branches.
A report is included.

## Overview

An implementation of morfological operators, in order to compare it with the sequential C++ version ([here](https://github.com/rickie95/MorphOpsCPP)) and measure speedup.

Code works with .ppm B/W images, extracting only the first channel if multiple channels are detected. Results are also .ppm images, timings are always saved in a .csv file.

Regarding operators, multiple shapes are avaiable: you can even set a custom sizes. Check out StructuringElement.h.

MorphableOperators.h contains implementations of some well know tranformations: erosion, dilatation, opening, closing, top-hat and bottom-hat transform. Derived transformations were implemented calling base transformations methods.

## Versions
Every version stores the structuring element in constant memory.

_*master*_: branch use shared memory for store locally each tile, compute the transformation on that tile and writes back in global memory. Tile's paddin is managed by two threads.

_*optimized-padding*_: works as above, except for padding's loading: this time all threads on the border work to load the padding.

_*no-shared-memory*_: pretty explicative, works without shared memory. Every read and write operation is done on global memory.

## Results

GPU timings were recorded on a Nvidia K80, CPU timings on a AMD Ryzen 1600; every GPU version was tested varing input tile's size from 8 to 32.
5 images from 400x400 5000x5000 were used for the test, below a table with three images and tile's size set to 16, full data can be found on the report.

Timings are in seconds.

Img res|Operation|CPU|master_16|optimized-padding_16|no-shared-memory_16|
-------|---------|---|---------|--------------------|-------------------|
432x596|Erosion     |0,0188|0,0005|0,0004|0,0003|
432x596|Dilatation  |0,0380|0,0011|0,0008|0,0006|
995x1134| Erosion   |0,4960|0,0142|0,0112|0,0082|
995x1134|Dilatation |1,0288|0,0284|0,0225|0,0164|
4871x6466|Erosion   |2,0761|0,0584|0,0464|0,0339|
4871x6466|Dilatation|4,1218|0,1180|0,0935|0,0678|
