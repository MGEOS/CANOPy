"""
Voxel-based ray tracing and occlusion mapping.
--------------------------
author: Matthias Gassilloud
date: 03.06.2025
--------------------------
This module implements algorithms for voxel-based occlusion mapping of LiDAR data.
A ray tracing of LIDAR beams through voxel space is performed and voxels are classified
in four categories: unobserved, occluded, empty, and filled. The module also provides
functionality for normalizing voxel heights using a digital terrain model (DTM).

The algorithms are an improved version of Gassilloud et al. (2025) [1]. The docstrings were
partially completed by the Claude 3.7 Sonnet large language model (Anthropic, 2024).

References:

[1] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""

### import modules
import sys
import os
import time
import numpy as np
import numpy.ma as ma
from osgeo import gdal
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


### include modules from parent directories
from pathlib import Path
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from CANOPy.geos_utils.geodata_tb.point_cloud_tb import normalize_vox_array, voxelize_pointcloud
from CANOPy.geos_utils.geodata_tb.raster_tb import read_raster_array, crop_raster, write_raster_array, transform_affine_from_origin
from CANOPy.geos_utils.algorithms_tb.voxel_traversal.ray_vox_trav_3d_nb import ray_vox_trav_nb
from CANOPy.geos_utils.numba_tb.numba_tb import nb_float_to_string
from CANOPy.geos_utils.plotting_tb.plotting_tb import change_spine, set_size
from CANOPy.occlusion_mapping.configs.config import create_occlusion_mapping_config
from CANOPy.occlusion_mapping.pulse_origin_reconstruction import extend_trajectory_to_height


def top_view(vox_array, cfg, classification, normalize=True):
    '''calculate class percentage for top view'''

    if normalize:
        '''percentage below first filled or occluded voxel'''
        first_idx = np.argmax(np.logical_or(vox_array == cfg["voxel_traversal_kwargs"]["classification_values"]["filled"],
                                            vox_array == cfg["voxel_traversal_kwargs"]["classification_values"]["occluded"]), axis=0)
        first_idx = vox_array.shape[0] - first_idx  # reverse
        top = np.sum(vox_array == cfg["voxel_traversal_kwargs"]["classification_values"][f"{classification}"], axis=0)
        return (top/ first_idx) * 100

    else:
        return np.mean(vox_array == cfg["voxel_traversal_kwargs"]["classification_values"][f"{classification}"], axis=0) * 100  # not normalized

def side_view(vox_array, cfg, classification):
    '''west_east side view'''
    return np.mean(vox_array == cfg["voxel_traversal_kwargs"]["classification_values"][f"{classification}"], axis=1) * 100  # not normalized

def plot_example(vox_array, xyz_bounds, cfg, classification="occluded", normalized=False, write=False):


    ### create title/file string
    if normalized:
        norm_str = "_normalized"
    else:
        norm_str = ""


    ### plot parameters
    titlesize = 14
    ax_label_size = 12
    ax_width = 1

    ax_tick_label_size = 8
    ax_tick_width = 1
    ax_tick_length = 4

    cbar_labelsize = 12
    cbar_ticklabelsize = 10


    ### colormap
    cmap = mpl.colormaps['magma_r']
    cbar_padding = 0.05  # this is percent (times the main Axes height)
    cbar_size = 0.1  # this is percent (times image witdh)


    ### figure top view
    im_ratio = vox_array.shape[1] / (vox_array.shape[2] * (1+cbar_size))  # ratio to match northing
    
    fig, (ax1, axc1) = plt.subplots(1, 2,
        figsize=set_size(ratio=im_ratio),
        layout="constrained",
        gridspec_kw={'width_ratios': [1, cbar_padding*4]})

    im1 = ax1.imshow(top_view(vox_array, cfg, classification, normalize=False),
        extent=[xyz_bounds[0,0], xyz_bounds[1,0], xyz_bounds[0,1], xyz_bounds[1,1]],
        vmin = 0, vmax=100,
        cmap=cmap)

    # axis coordinates
    ax1.ticklabel_format(axis="both", style="plain", useOffset=False)  # tik label format
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks

    # axis parameters
    ax1.tick_params(axis='both', which='major', labelsize=ax_tick_label_size, width=ax_tick_width, length=ax_tick_length)
    ax1.set_xlim(xyz_bounds[0,0], xyz_bounds[1,0])  # x
    ax1.set_ylim(xyz_bounds[0,1], xyz_bounds[1,1])  # y
    change_spine(ax1, width=ax_width)

    # labels
    ax1.set_xlabel("Easting", fontsize=ax_label_size)
    ax1.set_ylabel("Northing", fontsize=ax_label_size)
    ax1.set_title(f"Top View ({classification}{norm_str})", fontsize=titlesize)


    ### separate ax for colorbar
    axc1.axis('off')  # workaround for padding on right side

    # colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_padding)
    cbar = fig.colorbar(im1, cax=cax, ax=ax1)
    cbar.ax.tick_params(labelsize=cbar_ticklabelsize, width=ax_tick_width, length=ax_tick_length) 
    cbar.set_label("Percentage of voxels (%)", size=cbar_labelsize)


    if write:
        fpath = os.path.join(cfg["root_dir"], "voxel_classification", f"top_view_{classification}{norm_str}.png")
        fig.savefig(fpath, dpi=300)

    ### plot
    plt.show(block=False)




    ### figure side view
    im_ratio = (vox_array.shape[0] ) / (vox_array.shape[2] * (1+cbar_size))  # ratio to match height
     
    fig, (ax1, axc1) = plt.subplots(1, 2,
        figsize=set_size(ratio=im_ratio),
        layout="constrained",
        gridspec_kw={'width_ratios': [1, cbar_padding*4]})

    im1 = ax1.imshow(side_view(vox_array, cfg, classification),
        extent=[xyz_bounds[0,0], xyz_bounds[1,0], xyz_bounds[0,2], xyz_bounds[1,2]],
        vmin = 0, vmax=100,
        cmap=cmap)

    # axis coordinates
    ax1.ticklabel_format(axis="both", style="plain", useOffset=False)  # tik label format
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks

    # axis parameters
    ax1.tick_params(axis='both', which='major', labelsize=ax_tick_label_size, width=ax_tick_width, length=ax_tick_length)
    ax1.set_xlim(xyz_bounds[0,0], xyz_bounds[1,0])  # y
    ax1.set_ylim(xyz_bounds[0,2], xyz_bounds[1,2])  # z
    change_spine(ax1, width=ax_width)

    # labels
    ax1.set_xlabel("Easting", fontsize=ax_label_size)
    ax1.set_ylabel("Height", fontsize=ax_label_size)
    ax1.set_title(f"West-East View ({classification}{norm_str})", fontsize=titlesize)


    ### separate ax for colorbar
    axc1.axis('off')  # workaround for padding on right side

    # colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_padding)
    cbar = fig.colorbar(im1, cax=cax, ax=ax1)
    cbar.ax.tick_params(labelsize=cbar_ticklabelsize, width=ax_tick_width, length=ax_tick_length) 
    cbar.set_label("Percentage of voxels (%)", size=cbar_labelsize)


    if write:
        fpath = os.path.join(cfg["root_dir"], "voxel_classification", f"side_view_{classification}{norm_str}.png")
        fig.savefig(fpath, dpi=300)

    ### plot
    plt.show(block=False)



    ### figure sliced side view
    fig, (ax1, axc1) = plt.subplots(1, 2,
        figsize=set_size(ratio=im_ratio),
        layout="constrained",
        gridspec_kw={'width_ratios': [1, cbar_padding*4]})
    
    im1 = ax1.imshow(side_view(vox_array[:,150:180,:], cfg, classification),
        extent=[xyz_bounds[0,0], xyz_bounds[1,0], xyz_bounds[0,2], xyz_bounds[1,2]],
        vmin = 0, vmax=100,
        cmap=cmap)

    # axis coordinates
    ax1.ticklabel_format(axis="both", style="plain", useOffset=False)  # tik label format
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))  # number of ticks

    # axis parameters
    ax1.tick_params(axis='both', which='major', labelsize=ax_tick_label_size, width=ax_tick_width, length=ax_tick_length)
    ax1.set_xlim(xyz_bounds[0,0], xyz_bounds[1,0])  # y
    ax1.set_ylim(xyz_bounds[0,2], xyz_bounds[1,2])  # z
    change_spine(ax1, width=ax_width)

    # labels
    ax1.set_xlabel("Easting", fontsize=ax_label_size)
    ax1.set_ylabel("Height", fontsize=ax_label_size)
    ax1.set_title(f"West-East Sliced View ({classification}{norm_str})", fontsize=titlesize)


    ### separate ax for colorbar
    axc1.axis('off')  # workaround for padding on right side

    # colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_padding)
    cbar = fig.colorbar(im1, cax=cax, ax=ax1)
    cbar.ax.tick_params(labelsize=cbar_ticklabelsize, width=ax_tick_width, length=ax_tick_length) 
    cbar.set_label("Percentage of voxels (%)", size=cbar_labelsize)


    if write:
        fpath = os.path.join(cfg["root_dir"], "voxel_classification", f"side_sliced_view_{classification}{norm_str}.png")
        fig.savefig(fpath, dpi=300)

    ### plot
    plt.show(block=False)

def occlusion_classification(cfg):
    """Occlusion classification of a 3D voxel space.
    
    This function builds LiDAR beam trajectories, applies ray tracing and classifies
    voxels into four categories: unobserved, occluded, empty, and filled. The
    classification is based on voxel traversal of both observed rays (from pulse emission
    to last return) and occluded rays (from last return to ground).
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing parameters for voxel traversal and
        classification, including:
        - voxel_traversal_kwargs.params.cell_size: Size of each voxel cell
        - voxel_traversal_kwargs.params.boundary: Spatial boundary for voxelization
        - voxel_traversal_kwargs.params.nb_cell: Number of cells in each dimension
        - voxel_traversal_kwargs.classification_values: Classification encoding values
        - rays_path: Path to file containing ray start and end coordinates
        - point_cloud_path: Path to the .las point cloud file
        - fitler_rays_by_location: Boolean flag for filtering rays by location
        - fitler_rays_by_location_kwargs: Parameters for ray filtering
        
    Returns
    -------
    numpy.ndarray
        3D array of shape (nz, ny, nx) where each element contains a classification
        value according to cfg.voxel_traversal_kwargs.classification_values.
    numpy.ndarray
        Array of shape (2, 3) containing the [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        boundaries of the voxel space.
        
    Notes
    -----
    The classification process involves several steps:
    1. Loading ray trajectories from the rays_path file
    2. Optionally filtering rays by location using a polygon
    3. Extending ray trajectories to ground level to model occlusion
    4. Applying voxel traversal to observed and occluded rays
    5. Voxelizing the point cloud to identify filled voxels
    6. Combining these results to produce the final classification
    
    The classification values are:
    - unobserved: Voxels not traversed by any ray
    - occluded: Voxels traversed by occluded rays only
    - empty: Voxels traversed by observed rays (and possibly occluded rays)
    - filled: Voxels containing at least one point from the point cloud
    """

    ### read cfg
    cell_size = cfg["voxel_traversal_kwargs"]["params"]["cell_size"]
    xyz_bounds = np.array(cfg["voxel_traversal_kwargs"]["params"]["boundary"])
    nb_cell = np.array(cfg["voxel_traversal_kwargs"]["params"]["nb_cell"])

    
    ### read trajectory start and end point
    with open(cfg["rays_path"], 'rb') as f:
        start_coordinate = np.load(f)
        end_coordinate = np.load(f)


    ### checks
    assert start_coordinate.shape == end_coordinate.shape  # check if equal number of instances
    assert start_coordinate.shape[1] == 3  # check if 3 coordinates


    ### filter rays by area
    if cfg["fitler_rays_by_location"]:
        from CANOPy.geos_utils.geodata_tb.point_cloud_tb import crop_point_cloud
        mask_idx, stats = crop_point_cloud(start_coordinate[:,:2], cfg["fitler_rays_by_location_kwargs"]["filter_polygon_path"], status=True)
        start_coordinate = start_coordinate[mask_idx]
        end_coordinate = end_coordinate[mask_idx]

    
    ### reconstruct occlusion end coordiante
    occlusion_coordinate = extend_trajectory_to_height(start_coordinate, end_coordinate, 0)


    ### [ OPTIONAL ] calculate factor to reduce floating point precision error (e.g. for cell_size = 0.1)
    # take care, when passing factored valeus to voxel traversal, factored boundaries will be returned -> divide boundary resulty by factor
    decimals_cell_size = nb_float_to_string(cell_size)[::-1].find('.')  # number of float decimals

    if decimals_cell_size == -1:
        decimals_cell_size = 0

    # factor
    factor = 10**decimals_cell_size

    # scale bounds
    cell_size_f = np.round(cell_size * factor)
    xyz_bounds_f = np.round(xyz_bounds * factor)

    # scale coordinates
    start_coordinate *= factor  # replace
    end_coordinate *= factor  # replace
    occlusion_coordinate *= factor  # replace



    ### numba compilation
    print("Numba compilation of voxel traversal algorithm, test run...")
    start = time.time()
    _, boundary, stats = ray_vox_trav_nb(rays=np.stack((start_coordinate, end_coordinate),axis=1)[:1], boundary=xyz_bounds_f, cell_size=cell_size_f)
    end = time.time()
    print(f"    Elapsed time (with compilation): {round((end - start)/ 60, 4)} min")
    
    storage, stats = None, None  # close
    

    ### voxel ray tracing observed rays
    print("Voxel traversal of rays observed...")
    start = time.time()
    vox_observed, boundary_observed, stats_observed = ray_vox_trav_nb(rays=np.stack((start_coordinate, end_coordinate), axis=1), boundary=xyz_bounds_f, cell_size=cell_size_f)
    end = time.time()
    print(f"    Elapsed time (without compilation): {round((end - start)/ 60, 4)} min")
    
    start_coordinate = None  # close


    ### voxel ray tacing occluded rays
    print("Voxel traversal of rays occluded...")
    start = time.time()
    vox_occluded, boundary_occluded, stats_occluded = ray_vox_trav_nb(rays=np.stack((end_coordinate, occlusion_coordinate), axis=1), boundary=xyz_bounds_f, cell_size=cell_size_f)
    end = time.time()
    print(f"    Elapsed time (without compilation): {round((end - start)/ 60, 4)} min")


    ### checks        
    assert np.array_equal(boundary_observed, boundary_occluded)  # assert boundary equal
    assert np.array_equal(boundary_observed, xyz_bounds_f)   # test--- if eveything works fine one could put add nb_cell directly as input and discard return  of boundary / nb_cell
    
    end_coordinate, occlusion_coordinate = None, None  # close


    ### voxelize point cloud
    print("Voxelize point cloud...")
    start = time.time()
    vox_filled = voxelize_pointcloud(cfg["point_cloud_path"], xyz_bounds, cell_size, nb_cell)
    end = time.time()
    print(f"    Elapsed time: {round((end - start)/ 60, 4)} min")


    ### classify result
    cfg_classification = cfg["voxel_traversal_kwargs"]["classification_values"]
    storage_classification = np.full((nb_cell[2], nb_cell[1], nb_cell[0]), cfg_classification["unobserved"], dtype=np.byte)
    storage_classification[vox_occluded == 1] = cfg_classification["occluded"]
    storage_classification[vox_observed == 1] = cfg_classification["empty"]
    storage_classification[vox_filled == 1] = cfg_classification["filled"]
    

    ### invert index
    storage_classification = storage_classification[::-1,::-1,:]  # revert z, y index

    
    print("Occlusion classification complete.")
    return storage_classification, boundary / factor

def main():

        
    ### argument parser
    parser = argparse.ArgumentParser(description='Voxel based occlusion mapping.')
    parser.add_argument('--config',
                        help='Config file for occlusion mapping')
    args = parser.parse_args()


    ### create config
    cfg = create_occlusion_mapping_config(args.config)
    metadata = {}


    ### read cfg parameters
    cell_size = cfg["voxel_traversal_kwargs"]["params"]["cell_size"]
    xyz_bounds = cfg["voxel_traversal_kwargs"]["params"]["boundary"]
    nb_cell = cfg["voxel_traversal_kwargs"]["params"]["nb_cell"]
    print(f"XYZ Boundaries: {xyz_bounds}, Number of cells: {nb_cell}")  # status

    use_aoi_mask = cfg["aoi"] and cfg["aoi_kwargs"]["aoi_polygon"]["mask"]
    normalize_height = cfg["normalize_height"]
    


    ### occlusion classification
    vox_ray_class, xyz_bounds = occlusion_classification(cfg)


    ### create aoi mask
    if use_aoi_mask:  # mask dtm with polygon
    
        # initialize bool raster
        mask_aoi_array = np.full((nb_cell[1],nb_cell[0]), 1, dtype = np.byte)  # create mask array, initialize True
        mask_aoi_transform = transform_affine_from_origin(xyz_bounds[0][0], xyz_bounds[1][1], cell_size, cell_size)  # create transform
        write_raster_array(cfg["aoi_kwargs"]["array_mask"], mask_aoi_array, cfg["epsg_code"], mask_aoi_transform)  # store as raster

        # mask bool raster with "0" outside polygon
        crop_raster(raster_in= cfg["aoi_kwargs"]["array_mask"],
                        raster_out= cfg["aoi_kwargs"]["array_mask"],  # overwrite
                        crop_shp= cfg["aoi_kwargs"]["aoi_polygon"]["aoi_path"],  # shapefiel to mask raster
                        crop= False)  # mask raster with "0" instead of cropping

        # read bool raster, "0" = masked 
        aoi_mask, aoi_mask_meta = read_raster_array(cfg["aoi_kwargs"]["array_mask"])  # read aoi mask
        aoi_mask = ~aoi_mask.astype(bool)  # now "True = masked"

        # mask
        vox_ray_class = ma.masked_array(vox_ray_class, mask=np.broadcast_to(aoi_mask, vox_ray_class.shape))  # A broadcast usually does not involve copying the data n-time. It just simulates it by setting a stride appropriately.


    ### store result
    np.savez(cfg['voxel_result'], vox_ray_class)
    
    metadata["xyz_bounds"] = xyz_bounds.tolist()
    metadata["cell_size"] = cell_size
    
    with open(cfg['metadata_result'], 'w') as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)


    ### plot example
    plot_example(vox_array=vox_ray_class, xyz_bounds=xyz_bounds, cfg=cfg, classification="unobserved", write=True)
    plot_example(vox_array=vox_ray_class, xyz_bounds=xyz_bounds, cfg=cfg, classification="occluded", write=True)
    plot_example(vox_array=vox_ray_class, xyz_bounds=xyz_bounds, cfg=cfg, classification="empty", write=True)
    plot_example(vox_array=vox_ray_class, xyz_bounds=xyz_bounds, cfg=cfg, classification="filled", write=True)
    

    ### match DTM to regular voxel grid for height normalization
    if normalize_height:

        # reproject/ match voxel grid / crop to xyz bounds, documentation: https://gdal.org/en/stable/api/python/utilities.html
        ds = gdal.Warp(cfg["normalize_height_kwargs"]['dtm_tmp_path'],
                    cfg["normalize_height_kwargs"]["dtm_path"],
                    outputBounds= (xyz_bounds[0,0], xyz_bounds[0,1], xyz_bounds[1,0], xyz_bounds[1,1]),  # (minX, minY, maxX, maxY) in target SRS
                    xRes= cell_size,  # cell_size
                    yRes= cell_size,  # cell_size
                    width= nb_cell[0],  # cols
                    height= nb_cell[1],  # rows
                    dstSRS= f"EPSG:{cfg['epsg_code']}",  # srs
                    resampleAlg= "nearest"  # can be adapted to the users needs: resampleAlg = nearest|bilinear|cubic|cubicspline|lanczos|average|rms|mode|min|max|med|q1|q3|sum (default: nearest)
                    )
        ds = None  # close

        # read dtm
        dtm, dtm_meta = read_raster_array(cfg["normalize_height_kwargs"]['dtm_tmp_path'])  # read aoi mask
        dtm_nodata = dtm_meta['nodata']  # raster no data value            


        ### if aoi polygon mask check only for no data values within aoi
        if use_aoi_mask:  # mask dtm with polygon
        
            # check if no data value are within aoi
            dtm_missing_aoi = (dtm == dtm_nodata) & ~aoi_mask  # (True where no data AND True where within AOI)

            if np.any(dtm_missing_aoi):  # True if at least one True
                            
                print(f"Existing DTM no data value: {dtm_meta['nodata']}, assert this aligns with DTM values.")
        
                raise ValueError(f"Missing {np.sum(dtm_missing_aoi)} DTM grid cell values within AOI. " \
                "Values necessary for height normalization.\n" \
                "  1) Assert AOI Polygon is covered by DTM.\n" \
                "  2) Check if DTM has missing values within Polygon.\n" \
                "  -> You could try an interpolation to fill missing DTM values.")

            else:  # mask array outside aoi polygon with np.nan

                dtm[dtm == dtm_nodata] = np.nan  # set to np.nan
                dtm[aoi_mask] = np.nan  # set values outside aoi to np.nan


        else: # no masking (dtm already matched to voxel grid dimensions), still check if no data in xyz bounds
            
            # check if no data value are within xyz bounds
            dtm_missing_xyz_bounds = (dtm == dtm_nodata)  # True where no data 
            
            if np.any(dtm_missing_xyz_bounds):  # True if at least one True

                print(f"Existing DTM no data value: {dtm_meta['nodata']}, assert this aligns with DTM values.")
        
                raise ValueError(f"Missing {np.sum(dtm_missing_xyz_bounds)} DTM grid cell values in xyz bounds. " \
                "Values necessary for height normalization.\n" \
                "  1) Assert xyz bounds are covered by DTM.\n" \
                "  2) Check if DTM has missing values within xyz bounds.\n" \
                "  -> You could try an interpolation to fill missing DTM values.")

    
        ### normalize height
        vox_ray_class_norm, xyz_bounds_norm = normalize_vox_array(vox_ray_class, dtm, xyz_bounds, cell_size)


        ### if aoi polygon mask check only for no data values within aoi
        if use_aoi_mask:  # mask dtm with polygon
            vox_ray_class_norm = ma.masked_array(vox_ray_class_norm, mask=np.broadcast_to(aoi_mask, vox_ray_class_norm.shape))


        ### store
        np.savez(cfg['voxel_result_normalized_height'], vox_ray_class_norm)

        metadata["xyz_bounds_norm"] = xyz_bounds_norm.tolist()
        
        with open(cfg['metadata_result'], 'w') as fp:
            json.dump(metadata, fp, sort_keys=True, indent=4)


        ### plot example
        plot_example(vox_array=vox_ray_class_norm, xyz_bounds=xyz_bounds_norm, cfg=cfg, classification="unobserved", normalized=True, write=True)
        plot_example(vox_array=vox_ray_class_norm, xyz_bounds=xyz_bounds_norm, cfg=cfg, classification="occluded", normalized=True, write=True)
        plot_example(vox_array=vox_ray_class_norm, xyz_bounds=xyz_bounds_norm, cfg=cfg, classification="empty", normalized=True, write=True)
        plot_example(vox_array=vox_ray_class_norm, xyz_bounds=xyz_bounds_norm, cfg=cfg, classification="filled", normalized=True, write=True)
        

        print("Occlusion mapping complete.")  # status

if __name__ == "__main__":
    main()