import sys
import os
import warnings
import yaml
from pathlib import Path
import numpy as np
from rasterio.crs import CRS

current_dir = Path(__file__).parent.parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from CANOPy.geos_utils.geodata_tb.vector import polygon_bounds
from CANOPy.geos_utils.data_management.data_management_tb import check_file_exists, check_dir_exists, mkdir_if_missing
from CANOPy.geos_utils.geodata_tb.point_cloud_tb import point_clouds_xyz_range
from CANOPy.geos_utils.algorithms_tb.voxel_traversal.ray_vox_trav_3d_nb import vox_aoi


def create_sensor_position_reconstruction_config(config_file):


    ### read yml   
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = {}
   
    for k, v in config.items():  # copy
        cfg[k] = v


    ### check files
    check_dir_exists(cfg["root_dir"])
    check_file_exists(cfg["point_cloud_path"])


    ### check crs args
    try:
        crs = CRS.from_epsg(cfg['epsg_code'])
    except:
        print(f"{cfg['epsg_code']} does not seem to be valid")

    try:
        unit = crs.linear_units
        if not unit.lower() in ('metre', 'meter', 'm'):
            raise Exception
    except:
        print(f"Unit: '{unit}' of crs needs to be metric.")
        

    ### path management
    base_dir = os.path.join(cfg["root_dir"], "sensor_position_reconstruction")
    mkdir_if_missing(base_dir)
    cfg['position_reconstruction'] = os.path.join(base_dir, 'sensor_position_reconstruction.gpkg')


    ### check reconstruction args
    if cfg["sensor_position_reconstruction_kwargs"]["positions_per_second"] <= 0:
        raise ValueError("Positions per second cannot be <= 0.")
    
    if cfg["sensor_position_reconstruction_kwargs"]["positions_per_second"] > 200:
        warnings.warn("Ensure you have enough trajectories per second to reconstruct > 200 sensor positions per second.")

    if cfg["sensor_position_reconstruction_kwargs"]["traj_number_min"] <= 0:
        raise ValueError("Minimum trajectories cannot be <= 0.")
    
    if cfg["sensor_position_reconstruction_kwargs"]["distance_max"] < 0:
        raise ValueError("Maximum distance between trajectory and reconstructed sensor position cannot be < 0.")

    if 0 < cfg["sensor_position_reconstruction_kwargs"]["distance_max"] <= 0.001:
        warnings.warn("Very small maximum distance between trajectory and reconstructed sensor position, some positions might be filtered out.")


    return cfg

def create_pulse_origin_reconstruction_config(config_file):


    ### read yml   
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = {}
   
    for k, v in config.items():  # copy
        cfg[k] = v


    ### check files
    check_dir_exists(cfg["root_dir"])
    check_file_exists(cfg["point_cloud_path"])
    check_file_exists(cfg["sensor_position_path"])


    ### path management
    base_dir = os.path.join(cfg["root_dir"], "pulse_origin_reconstruction")
    mkdir_if_missing(base_dir)
    cfg['rays_path'] = os.path.join(base_dir, 'rays.npy')


    return cfg

def create_occlusion_mapping_config(config_file):


    ### read yml   
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    cfg = {}
   
    for k, v in config.items():  # copy
        cfg[k] = v


    ### check files
    check_dir_exists(cfg["root_dir"])
    check_file_exists(cfg["rays_path"])
    check_file_exists(cfg["point_cloud_path"])


    ### check crs args
    try:
        crs = CRS.from_epsg(cfg['epsg_code'])
    except:
        print(f"{cfg['epsg_code']} does not seem to be valid")

    try:
        unit = crs.linear_units
        if not unit.lower() in ('metre', 'meter', 'm'):
            raise Exception
    except:
        print(f"Unit: '{unit}' of crs needs to be metric.")
        

    ### path management
    base_dir = os.path.join(cfg["root_dir"], "voxel_classification")
    mkdir_if_missing(base_dir)
    cfg['voxel_result'] = os.path.join(base_dir, 'result.npz')
    cfg['metadata_result'] = os.path.join(base_dir, 'metadata_result.json')


    ### check aoi args
    if cfg["aoi"]:

        ### bounds: check if xyz_bounds provided
        if cfg["aoi_kwargs"]["xyz_bounds"] is not None:
            cfg["aoi_kwargs"]["xyz_bounds"] = np.array(cfg["aoi_kwargs"]["xyz_bounds"])  # convert to np.array
            print("xyz_bounds:", cfg["aoi_kwargs"]["xyz_bounds"])

        else:  # calculate xyz bounds from aoi
            check_file_exists(cfg["aoi_kwargs"]["aoi_polygon"]["aoi_path"]) 
            aoi_bounds = polygon_bounds(cfg["aoi_kwargs"]["aoi_polygon"]["aoi_path"])  #  get xy value from AOI
            _, _, _, _, z_min, z_max = point_clouds_xyz_range(las_paths=[cfg["point_cloud_path"]])  # get z value

            xyz_bounds = np.array([
                [aoi_bounds[0], aoi_bounds[1], np.min(z_min)],
                [aoi_bounds[2], aoi_bounds[3], np.max(z_max)]
                ])
            cfg["aoi_kwargs"]["xyz_bounds"] = xyz_bounds
        
        ### mask
        if cfg["aoi_kwargs"]["aoi_polygon"]["mask"]:
            check_file_exists(cfg["aoi_kwargs"]["aoi_polygon"]["aoi_path"])
            cfg["aoi_kwargs"]["array_mask"] = os.path.join(base_dir, 'array_mask.tif')   
            

    else:  ### calculate xyz_bounds from point cloud
        
        x_min, x_max, y_min, y_max, z_min, z_max = point_clouds_xyz_range(las_paths=[cfg["point_cloud_path"]])  # get z value

        xyz_bounds = np.array([
            [np.min(x_min), np.min(y_min), np.min(z_min)],
            [np.max(x_max), np.max(y_max), np.max(z_max)]
            ])
        cfg["aoi_kwargs"]["xyz_bounds"] = xyz_bounds


    ### check vox trav args
    cell_size = cfg["voxel_traversal_kwargs"]["params"]["cell_size"]
    if cell_size <= 0:
        raise ValueError("Cell size cannot be 0.")


    ### calculate number of cells
    boundary, nb_cell = vox_aoi(xyz_bounds, cfg["voxel_traversal_kwargs"]["params"]["cell_size"])  # store in cfg
    cfg["voxel_traversal_kwargs"]["params"]["boundary"] = boundary #.tolist()
    cfg["voxel_traversal_kwargs"]["params"]["nb_cell"] = nb_cell #.tolist()


    ### check filter args
    if cfg["fitler_rays_by_location"]:
        check_file_exists(cfg["fitler_rays_by_location_kwargs"]["filter_polygon_path"])


    ### check height normalization args
    if cfg["normalize_height"]:
        check_file_exists(cfg["normalize_height_kwargs"]["dtm_path"])
        cfg["normalize_height_kwargs"]['dtm_tmp_path'] = os.path.join(base_dir, 'dtm_tmp.tif')        
        cfg['voxel_result_normalized_height'] = os.path.join(base_dir, 'result_normalized_height.npz')


    return cfg
