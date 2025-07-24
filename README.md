# CANOPy - Canopy Attentuation and Occlusion in Python

The CANOPy module contains various usefull point cloud data processing functions in python.


## Tools
1. [occlusion mapping](./occlusion_mapping/tutorial.ipynb)
    * perform occlusion mapping using solely a point cloud
2. [geospatial utils](./geos_utils)
    * collection of various geospatial processing and data management tools, included as a submodule



## Dependencies
The CANOPy package requires the GEOS_utils repository as a submodule.



## Installation
Recommended to use python=3.10 or higher.

```bash
conda create -n canopy python=3.10
conda activate canopy
conda install numpy numba laszip laspy lazrs-python fiona shapely rasterio pyproj pandas geopandas gdal ipykernel pyyaml
```


## License
Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

