# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:06:55 2024

@author: uhljoha
@author: gochkat
"""
import geopandas as gp
import os
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import matplotlib.patches as mpatches
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import csv
import scipy.stats
import pylandstats
import seaborn as sns

print("GDAL version:", gdal.__version__)
print("GDAL_DATA:", os.environ.get("GDAL_DATA"))

# Use Arial font, size 11
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11
})

version = 'v1'
# landshp = r'C:\DATA\2025_LAUs\LAU_database_%s_54009.gpkg'%version
# datadir = r'O:\03_MISC\2025_built_dynamics\results_LAUs_%s'%version

# DIRECTORIES
datadir = r'C:\PROCESSING\2025_built_dynamics\results_LAU_%s'%version
if not os.path.exists(datadir):
    os.makedirs(datadir)

# INPUT DATA
raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif'
LAU_gp_input = r'C:\DATA\2025_LAU\AU_RG_01M_2011_3035_Poland.gpkg'
LAU_gp = os.path.join(datadir,'LAU_RG_01M_2011_54009_Poland_%s.gpkg'%version)

# PROCESSES
reproject_lau = False
rasterize_lau=False
calc_tertiles = False

calc_stats = False
calc_class_stats = False
calc_lsm=False
calc_agrm = False

generate_tables_PL = False
plot_charts_PL = False
panel_plot_lsm_PL=False

generate_class_table=False 
panel_plot_pop_PL = False
plot_maps_PL=False
plot_agr_maps_PL = False
series_plot_lsm_PL = False


strata_names = {
    1: 'LowBU_LowPop', 2: 'MidBU_LowPop', 3: 'HighBU_LowPop',
    4: 'LowBU_MidPop', 5: 'MidBU_MidPop', 6: 'HighBU_MidPop', 
    7: 'LowBU_HighPop', 8: 'MidBU_HighPop', 9: 'HighBU_HighPop' }

# We compute LAU metrics for each year
years=np.arange(1975,2021,5)
# years = np.delete(years, np.where(years == 1980)) # Check the 1980 GHSL input data
# We compute agreement between the first and the last observed year
agr_periods = [[1975, 2020]]

raster_template = raster_bu
    
def _validate_input(pred: np.ndarray, ref: np.ndarray) -> None:
    """Validate that inputs are NumPy arrays of the same shape."""
    if not isinstance(pred, np.ndarray) or not isinstance(ref, np.ndarray):
        raise TypeError("Both `pred` and `ref` must be NumPy arrays.")
    if pred.shape != ref.shape:
        raise ValueError(f"Arrays must have the same shape. Found {pred.shape} and {ref.shape}.")

def cont_jaccard(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Jaccard Index (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(np.maximum(pred, ref))
    return float(numerator / denominator) if denominator != 0 else np.nan

def cont_recall(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Recall (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(ref)
    return float(numerator / denominator) if denominator != 0 else np.nan

def cont_precision(pred: np.ndarray, ref: np.ndarray) -> float:
    """Continuous Precision (NaN-tolerant)."""
    _validate_input(pred, ref)
    numerator = np.nansum(np.minimum(pred, ref))
    denominator = np.nansum(pred)
    return float(numerator / denominator) if denominator != 0 else np.nan

def fscore(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-score given precision, recall, and beta (NaN-robust)."""
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    if beta <= 0:
        raise ValueError("`beta` must be greater than 0.")
    if precision == 0 and recall == 0:
        return np.nan
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

def RMSD(pred, ref) -> float:
    """ Root Mean Square Deviation (nan-proof) """
    _validate_input(pred, ref)
    diff_squared = (pred - ref) ** 2
    return np.sqrt(np.nanmean(diff_squared))

def MAD(pred, ref) -> float:
    """ Mean Absolute Deviation (nan-proof) """
    _validate_input(pred, ref)
    abs_diff = np.abs(pred - ref)
    return np.nanmean(abs_diff)

def MD(pred, ref) -> float:
    """ Mean Deviation (nan-proof) """
    _validate_input(pred, ref)
    return np.nanmean(pred) - np.nanmean(ref)

def MAPE(pred, ref) -> float:
    """ Mean Absolute Percentage Error (nan-proof) """
    _validate_input(pred, ref)
    # Only compute where ref > 0 and neither pred nor ref is NaN
    mask = (ref > 0) & (~np.isnan(ref)) & (~np.isnan(pred))
    if not np.any(mask):
        return np.nan  # or raise an error depending on use case
    mape = np.nanmean(np.abs((pred[mask] - ref[mask]) / ref[mask])) * 100
    return mape

def CR(arr1, arr2) -> float:
    """ Change Rate (nan-proof) """
    _validate_input(arr1, arr2)
    sum1 = np.nansum(arr1)
    sum2 = np.nansum(arr2)
    if sum1 == 0:
        return np.nan  # avoid division by zero
    return (sum2 - sum1) / sum1

def get_subset(bbox,currfile):
    # source: https://riptutorial.com/gdal/example/25844/read-subset-of-a-global-raster-defined-by-a-bounding-box
    ds = gdal.Open(currfile, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    # The inverse geotransform is used to convert lon/lat degrees to x/y pixel index
    inv_geotransform = gdal.InvGeoTransform(gt)            
    # Convert lon/lat degrees to x/y pixel for the dataset
    _x0, _y0 = gdal.ApplyGeoTransform(
        inv_geotransform, bbox[0], bbox[1])
    _x1, _y1 = gdal.ApplyGeoTransform(
        inv_geotransform, bbox[2], bbox[3])
    x0, y0 = min(_x0, _x1), min(_y0, _y1)
    x1, y1 = max(_x0, _x1), max(_y0, _y1)
    # Get subset of the raster as a numpy array
    data = band.ReadAsArray(int(x0), int(y0), int(x1-x0), int(y1-y0))
    nodataval = band.GetNoDataValue()
    data[data==nodataval]=0
    return data
    ds = None
    
if reproject_lau:
    # Load LAU source file
    shp = gp.read_file(LAU_gp_input)
    # Reproject shapefile to match raster CRS
    with rasterio.open(raster_template) as src:
        shp_r = shp.to_crs(src.crs)
    # Save reprojected LAU file to GPKG
    shp_r.to_file(LAU_gp, driver='GPKG')
    
if rasterize_lau:
    # Load shapefile
    shp = gp.read_file(LAU_gp)
    shp['lauid_num'] = np.arange(1, len(shp) + 1)
    
    # Extract geometry with LAU ID assigned
    with rasterio.open(raster_template) as src:
        # shp = shp.to_crs(src.crs)
        shapes = ((geom, value) for geom, value in zip(shp.geometry, shp.lauid_num))
        # Mask (crop) the raster template with shapefile geometry
        out_image, out_transform = mask(src, shp.geometry, crop=True)
        out_meta = src.meta.copy()
    
    # Update metadata for the cropped raster
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    # Use MemoryFile to temporarily hold the cropped raster
    with MemoryFile() as memfile:
        with memfile.open(**out_meta) as temp_raster:
            # The clipped template in memory is now the template raster
            # Rasterize your vector data to match this clipped template
            rasterized = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=(temp_raster.height, temp_raster.width),
                fill=0,
                transform=temp_raster.transform,
                all_touched=True,
                default_value=1,
                dtype=np.uint32
            )
    
            # Update metadata for output
            kwargs = out_meta.copy()
            kwargs.update({
                'dtype': 'uint32',
                'count': 1,
                'nodata': 0,
                'compress': 'lzw'
            })
    
            # Save to file
            output_path = os.path.join(datadir, 'ref-lau-2011-01m_%s.tif'%version)
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(rasterized, 1)

if calc_tertiles:
    print('Compute the tertiles values of bu an pop in a country in 2020')
    
    # Assign paths to BU and POP data
    rasters = {
        'BUILT_S': r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif',
        'POP': r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0\GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0.tif'}
    
    # Calculate tertile of built-up and population densities on country level
    shp = gp.read_file(LAU_gp)
    
    for country,countrydf in shp.groupby('CNTR_CODE'):
        # Union all geometries into a single one
        shp_union = countrydf.union_all(method='coverage')
        # Convert to GeoJSON-like format expected by rasterio
        mask_geom = [shp_union.__geo_interface__]
        # First, initialize empty outputs
        tertiles = {}
        arrays = {}
        for s in ['BUILT_S', 'POP']:
            raster_path = rasters[s]
            with rasterio.open(raster_path) as src:
                # Get the ratser data masked with the country boundaries
                out_image, out_transform = mask(src, mask_geom, crop=True)
                data = out_image[0]
                arrays[s] = data
                # Flatten and remove nodata and zero values
                data_flat = data[(data != src.nodata) & (data != 0)].flatten()
                # Compute tertiles 
                tertiles[s] = np.quantile(data_flat, [1/3, 2/3])
            
            out_meta = src.meta.copy()
            # Update metadata for the cropped raster
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
        
        # For POP replace values less than 1 with 1
        tertiles['POP'] = np.where(tertiles['POP'] < 1, 1, tertiles['POP'])
        # For both POP and BU round and check if all values are the same
        for key in tertiles.keys():
            tertiles[key] = np.round(tertiles[key], 0)
            if np.all(tertiles[key] == tertiles[key][0]):
                raise ValueError("All tertiles for 'POP' are the same. Cannot proceed.")


        # Save tertiles for both input datasets for each country as a csv
        output_path = os.path.join(datadir, 'tertiles_%s_2020_%s.csv'%(country, version))
        print('Save to CSV:',output_path)
        # Save to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value1', 'value2'])
            for key, arr in tertiles.items():
                writer.writerow([key] + list(arr))
        
if save_classified:
    rasters = {}
    for year in years:
        rasters['BUILT_S'] = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        rasters['POP'] = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year)) 
        
        shp = gp.read_file(LAU_gp)
        shp['landid_num']=np.arange(1,len(shp)+1)    
        shp[['xmin','ymin','xmax','ymax']]=shp.bounds
        total_overall=len(shp)
        counter_overall=0
        for country,countrydf in shp.groupby('CNTR_CODE'):           
            processdf = countrydf  
            total=len(processdf)   
            lsmdata=[]
            counter=0

            # Load the pre-computed tertiles for bu and pop arrays
            tertiles_path = os.path.join(datadir, 'tertiles_%s_2020_%s.csv'%(country, version))
            tertiles_d = {}
            with open(tertiles_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['key']
                    value1 = float(row['value1'])
                    value2 = float(row['value2'])
                    tertiles_d[key] = np.array([value1, value2])
                    
            # Create strata raster. First, initialize empty outputs
            strata_class = {}
            arrays = {}
            for s in ['BUILT_S', 'POP']:
                raster_path = rasters[s]
                with rasterio.open(raster_path) as src:
                    # Get the ratser data masked with the country boundaries
                    out_image, out_transform = mask(src, mask_geom, crop=True)
                    data = out_image[0]
                    arrays[s] = data
                    nd=src.nodata
                    
                terts = tertiles_d[s]
                # Create empty array of the same size as input data
                result = np.full(arrays[s].shape, np.nan)
                # Assign density classes
                result[arrays[s] <= terts[0]] = 0
                result[(arrays[s] > terts[0]) & (arrays[s] <= terts[1])] = 1
                result[arrays[s]> terts[1]] = 2
                # Mask out nodata values
                strata_class[s] = np.where(arrays[s]==src.nodata, np.nan, result)  
            
            pop_class = strata_class['POP']
            bu_class = strata_class['BUILT_S']
            
            strata = np.full(bu_class.shape, 0) #np.nan
            # Where both are valid (i.e., not NaN), assign combined class
            valid_mask = (~np.isnan(pop_class)) & (~np.isnan(bu_class))
            strata[valid_mask] = 1 + 3 * pop_class[valid_mask] + bu_class[valid_mask] 
           
            # Update metadata for output
            kwargs = out_meta.copy()
            kwargs.update({
                'dtype': 'uint32',
                'count': 1,
                'nodata': 0,
                'compress': 'lzw'
            })
    
            # Save to file
            output_path = os.path.join(datadir, 'classification_%s_%s_%s.tif'%(country, year, version))
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(strata, 1)
                        

if plot_classified_PL:
    country='PL'
    for year in years:
        # Open built-up array for the year
        raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        with rasterio.open(raster_path) as src:
            # Get the ratser data masked with the country boundaries
            out_image, out_transform = mask(src, mask_geom, crop=True)
            bu_data = out_image[0]
            nd=src.nodata
        # Path to the classified raster in native resolution    
        output_path = os.path.join(datadir, 'classification_%s_%s_%s.tif'%(country, year, version))
        # Reproject a file into the Polish projection (PUWG 1992)
        dst_crs = "EPSG:2180"
        # Open the file and reproject
        with rasterio.open(output_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })
        
            # Bufor na dane po reprojekcji
            strata = np.empty((height, width), dtype=np.float32)
            
            src_data = src.read(1)
            masked_data = np.where(bu_data > 0, src_data, np.nan)
            
        
            reproject(
                source=masked_data,
                destination=strata,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
            
        strata = np.where(strata > 0, strata, np.nan)

        # Define colors per class (1–9)
        strata_colors = {
        1: '#d9f0a3',  # light green
        2: '#78c679',  # medium green
        3: '#238443',  # dark green
        4: '#fff7bc',  # light yellow
        5: '#fec44f',  # orange-brown
        6: '#8c2d04',  # dark brown
        7: '#fee5d9',  # light red
        8: '#fcae91',  # medium red
        9: '#cb181d'   # dark red 
        }
        
        # Create a ListedColormap (for maps, imshow, etc.)
        cmap = mcolors.ListedColormap([strata_colors[i] for i in range(1, 10)])
   
        # Plot
        plt.figure(figsize=(6, 6), dpi=150)
        im = plt.imshow(strata, cmap=cmap, vmin=1, vmax=9)
        plt.text(1,1,str(year), fontsize=11)
        # plt.title("Built-up surface area and population stratification")
        plt.axis("off")
        
        # Legenda
        handles = [
            mpatches.Patch(color=cmap(i - 1), label=f"{strata_names[i]}")
            for i in range(1, 10)
        ]
        # plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5), title="Stratification")
        plt.tight_layout()
        # Save
        fig_path = os.path.join(datadir, "classification_%s_%s_%s.png"%(country, year, version))
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        
# Compute population, built-up sums for each LAU type
if calc_stats:
    for year in years:
        raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        raster_pop = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
    
        shp = gp.read_file(LAU_gp).to_crs("ESRI:54009")
        shp['landid_num']=np.arange(1,len(shp)+1)    
        shp[['xmin','ymin','xmax','ymax']]=shp.bounds
        total_overall=len(shp)
        counter_overall=0
        
        for country,countrydf in shp.groupby('CNTR_CODE'):
            
            processdf = countrydf
            total=len(processdf)   
            landdata=[]
            counter=0
                    
            for i,row in processdf.iterrows():
                counter+=1
                counter_overall+=1
                aname = row.LAU_NAME
                landid = row.landid_num
                
                try:
                    buarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_bu)  
                    # volarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_vol)  
                    poparr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_pop)  
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],os.path.join(datadir, 'ref-lau-2011-01m_%s.tif'%version))  
                except:
                    print('outside of domain')
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)
                                                  
                try:
                    curr_bu_bb_bin = buarr.copy()
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)  

                # Compute pop and bu stats
                try: 
                    bu = buarr.astype(float).copy()
                    bu[landarr!=landid]=np.nan 
                    # vol = volarr.astype(float).copy()
                    # vol[landarr!=landid]=np.nan 
                    pop = poparr.astype(float).copy()
                    pop[landarr!=landid]=np.nan 
                    
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if land areas are outside the raster data domain (eg overseas territories)  
                    
                # plt.imshow(bu)
                # plt.show()         
                
                if np.nansum(bu)==0:
                    print('NBU land?')            
                    continue
                      
                total_bu = np.nansum(bu)
                # total_vol = np.nansum(vol)
                total_pop = np.nansum(pop)       
        
                landdata.append([landid,aname,total_bu,total_pop])
                
                print(year,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                
            landdatadf=pd.DataFrame(landdata) 
            landdatadf.columns=['landid','name','total_bu','total_pop']
            
            landdatadf.to_csv(datadir+os.sep+'LAU_stats_%s_%s_%s.csv' %(country,year, version),index=False) 
    
    # Concatenate bu and pop stats   
    result_gdf = shp.copy()
    for country,countrydf in shp.groupby('CNTR_CODE'):
        for year in years:
            df = pd.read_csv(datadir+os.sep+'LAU_stats_%s_%s_%s.csv' %(country,year, version))
            # Select only the necessary columns for merge
            df_subset = df[["landid", 'total_bu', "total_pop"]].copy()
            df_subset = df_subset.rename(columns={
                "total_bu": 'bu_%s'%year,
                "total_pop": 'pop_%s'%year,
            })
            
            # Rename 'landid' to match 'result_gdf' key for direct merge, but without preserving it
            df_subset = df_subset.rename(columns={"landid": "landid_num"})
            
            # Merge without bringing in the 'landid' column explicitly
            result_gdf = result_gdf.merge(df_subset, how="left", on="landid_num")
    
    # Calculate area and diameter
    result_gdf["pow_km2"] = result_gdf.geometry.area / 1000000  # from m² to km²
    result_gdf["obwod_km"] = result_gdf.geometry.length / 1000   # from m to km
    # Save the final database to a GPKG
    result_gdf.to_file(datadir+os.sep+'LAU_%s_GHSL_54009_%s.gpkg'%(country,version),driver='GPKG')
    
if plot_maps_PL:
    # Load the final database with BU an POP counts per year in Poland
    country = 'PL'
    map_gdf = gp.read_file(datadir+os.sep+'LAU_%s_GHSL_54009_%s.gpkg'%(country,version))
    
    # map_gdf = gp.read_file(datadir+os.sep+'LAUs_mesoregions_GHSL_54009.gpkg')
    # Reproject the mad gdf to a Polish projection CRS Polkovo
    map_gdf = map_gdf.to_crs(2180)
    # Group polygons by the LAU polygon ID
    map_agg = (map_gdf.dissolve(
        by="LAU_NAME",
        aggfunc={
            "pop_1975": "sum",
            "pop_2020": "sum",
            "bu_1975": "sum",
            "bu_2020": "sum",
            "pow_km2": "sum"} ))
    # -------------------------------------------------------
    # Calculate % change in built-up area by polygon between 1975 and 2020
    # -------------------------------------------------------
    map_agg["bu_change_pct"] = ((map_agg["bu_2020"] - map_agg["bu_1975"]) /
                                map_agg["bu_1975"] * 100)
    
    # Classify into growth categories
    bu_bins = [-float("inf"), 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, float("inf")]
    # Matching class labels
    bu_labels = [
        "<50%", "50–75%", "75–100%", "100–125%", "125–150%", "150–175%", 
        "175–200%", "200–225%", "225–250%", "250–275%", "275–300%", ">300%"
    ]
       
    # Assign categories
    map_agg["bu_class"] = pd.cut(map_agg["bu_change_pct"], bins=bu_bins, labels=bu_labels, right=False)

    
    # -------------------------------------------------------
    # Calculate population density in 1975 and 2020
    # -------------------------------------------------------
    map_agg["pop_density_1975"] = map_agg["pop_1975"] / map_agg["pow_km2"]
    map_agg["pop_density_2020"] = map_agg["pop_2020"] / map_agg["pow_km2"]
    
    # Calculate % change in population density
    map_agg["pop_density_change_pct"] = ((map_agg["pop_density_2020"] - map_agg["pop_density_1975"]) /
                                         map_agg["pop_density_1975"] * 100)
    
    # Classify into change categories
    # Here we include possible decline category (<0%)
    pop_bins = [
    -float("inf"), -40, -25, -10, 5, 20, 35, 70, 120, 180, 240, 300, float("inf")
    ]
    
    pop_labels = [
        "< -40%", "-40% – -25%", "-25% – -10%", "-10% – 5%",
        "5% – 20%", "20% – 35%", "35% – 70%", "70% – 120%",
        "120% – 180%", "180% – 240%", "240% – 300%", "> 300%"
    ]

    map_agg["pop_class"] = pd.cut(map_agg["pop_density_change_pct"], bins=pop_bins, labels=pop_labels)
    
    # -------------------------------------------------------
    # Plot Figure 4.1: Built-up area change
    # -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    
    map_agg.plot(column="bu_class", cmap="Reds", legend=True, ax=ax[0], edgecolor="black", linewidth=0.1)
    ax[0].set_title("Change in built-up surface area 1975–2020", fontsize=14)
    ax[0].axis("off")
    
    # -------------------------------------------------------
    # Plot Figure 4.2: Population density change
    # -------------------------------------------------------
    map_agg.plot(column="pop_class", cmap="RdYlBu_r", legend=True, ax=ax[1], edgecolor="black", linewidth=0.1)
    ax[1].set_title("Change in population density 1975–2020", fontsize=14)
    ax[1].axis("off")
    
    plt.tight_layout()
    fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_pc_map_%s.png'%version,dpi=300,bbox_inches='tight')
    plt.show()

# if plot_charts_PL:
#     country='PL'
#     # Load the final database as a df
#     result_df = gp.read_file(datadir+os.sep+'LAU_%s_GHSL_54009_%s.gpkg'%(country,version),ignore_geometry=True)
       
#     # Group data by LAU type and aggregate sums
#     agg_dict = {f'pop_{year}': 'sum' for year in range(1975, 2025, 5)}
#     agg_dict.update({f'bu_{year}': 'sum' for year in range(1975, 2025, 5)})
#     agg_dict['pow_km2'] = 'sum'
#     agg_df = result_df.groupby('LAU_NAME').agg(agg_dict).reset_index()
    
#     # Calculate percentage changes
#     agg_df['pop_change_1975_2020_pct'] = (agg_df['pop_2020'] - agg_df['pop_1975']) / agg_df['pop_1975'] * 100
#     agg_df['bu_change_1975_2020_pct'] = (agg_df['bu_2020'] - agg_df['bu_1975']) / agg_df['bu_1975'] * 100
    
#     # Calculate population density 2020 (people per km2)
#     agg_df['pop_dens_2020_pp_km2'] = agg_df['pop_2020'] / agg_df['pow_km2']
#     agg_df['pop_dens_2020_pp_km2']  = agg_df['pop_dens_2020_pp_km2'].round(0).astype(int)
    
#     # Prepare table 4.1 (round values for clarity)
#     table_4_1 = agg_df[['LAU_NAME', 'pop_1975', 'pop_2020', 'pop_change_1975_2020_pct',
#                         'bu_1975', 'bu_2020', 'bu_change_1975_2020_pct', 'pop_dens_2020_pp_km2']].copy()
    
#     table_4_1['bu_1975'] = table_4_1['bu_1975'] / 1000  # convert from m to km
#     table_4_1['bu_2020'] = table_4_1['bu_2020'] / 1000  # similarly
#     table_4_1=table_4_1.rename(columns={"bu_1975": "bu_1975_km2","bu_2020": "bu_2020_km2"})
    
#     table_4_1 = table_4_1.round({
#         'pop_1975': 0,
#         'pop_2020': 0,
#         'pop_change_1975_2020_pct': 2,
#         'bu_1975_km2': 2,
#         'bu_2020_km2': 2,
#         'bu_change_1975_2020_pct': 2,
#         'pop_dens_2020_pp_km2': 2
#     })
#     table_4_1.to_csv(datadir + os.sep + 'BU_POP_change_1975-2020_%s.csv'%version, index=False)
#     table_4_1.to_clipboard(index=False)
#     print(table_4_1)
    
#     # --- Wykres 4.1 ---
    
#     # Calculate absolute population change (increase or decrease)
#     # agg_df['abs_pop_change_1975_2020'] = agg_df['pop_change_1975_2020_pct'].abs()
    
#     # Sort the entire DataFrame by absolute population change descending
#     plot_df = agg_df.sort_values(by='pop_change_1975_2020_pct', ascending=False)
    
#     # Calculate common y-axis limits based on min and max of both series
#     ymin = min(plot_df['pop_change_1975_2020_pct'].min(), plot_df['bu_change_1975_2020_pct'].min())
#     ymax = max(plot_df['pop_change_1975_2020_pct'].max(), plot_df['bu_change_1975_2020_pct'].max())
    
#     # Optionally add some padding
#     ymin -= (ymax - ymin) * 0.01
#     ymax += (ymax - ymin) * 0.1

#     # Plot population and built-up changes side by side
#     fig, ax1 = plt.subplots(figsize=(12,6))
    
#     bar_width = 0.35
#     index = range(len(plot_df))
    
#     # Bars for population change (%)
#     pop_bars = ax1.bar(index, plot_df['pop_change_1975_2020_pct'], bar_width, label='Zmiana liczby ludności 1975–2020 [%]', color='tab:blue')
    
#     # Bars for built-up change (%), shifted right
#     ax2 = ax1.twinx()
#     bu_bars = ax2.bar([i + bar_width for i in index], plot_df['bu_change_1975_2020_pct'],bar_width,label='Zmiana powierzchni terenów zabudowanych 1975–2020 [%]', color='tab:orange')
    
#     # Set same limits on both y-axes
#     ax1.set_ylim(ymin, ymax)
#     ax2.set_ylim(ymin, ymax)
    
#     # X-axis labels
#     ax1.set_xticks([i + bar_width/2 for i in index])
#     # ax1.set_xticklabels(plot_df['LAU_NAME'], rotation=45, ha='right')
#     ax1.set_xticklabels([])
    
#     # Labels and title
#     ax1.set_ylabel('Zmiana liczby ludności [%]')
#     ax2.set_ylabel('Zmiana powierzchni terenów zabudowanych [%]')
#     plt.title('Zmiana powierzchni terenów zabudowanych i liczby ludności w latach 1975-2020 w typach krajobrazu naturalnego')
    
#     # Legends
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')
    
#     plt.tight_layout()
#     fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_pc_bars_%s.png'%version,dpi=300,bbox_inches='tight')
#     plt.show()

#     ####### Make line plots  
#     # Columns for population and built-up area
#     pop_cols = [f"pop_{year}" for year in range(1975, 2025, 5)]
#     bu_cols = [f"bu_{year}" for year in range(1975, 2025, 5)]
    
#     # Compute relative values and final pop relative in 2020 for sorting
#     plot_data = []
#     for krajobraz in agg_df['LAU_NAME']:
#         pop_values = agg_df.loc[agg_df['LAU_NAME'] == krajobraz, pop_cols].values.flatten()
#         pop_relative = pop_values / pop_values[0] * 100
        
#         bu_values = agg_df.loc[agg_df['LAU_NAME'] == krajobraz, bu_cols].values.flatten()
#         bu_relative = bu_values / bu_values[0] * 100
        
#         plot_data.append({
#             "krajobraz": krajobraz,
#             "pop_relative": pop_relative,
#             "bu_relative": bu_relative,
#             "pop_2020": pop_relative[-1]  # for sorting
#         })
    
#     # Sort by population in 2020 relative to 1975
#     plot_data_sorted = sorted(plot_data, key=lambda x: x['pop_2020'], reverse=True)
    
#     # Create figure with two subplots
#     fig, (ax_bu, ax_pop) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    
#     years = range(1975, 2025, 5)
    
#     # Plot lines in sorted order
#     for data in plot_data_sorted:
#         ax_pop.plot(years, data['pop_relative'], label=data['krajobraz'], lw=2)
#         ax_bu.plot(years, data['bu_relative'], label=data['krajobraz'], lw=2)
    
#     # Titles and labels
#     ax_pop.set_title("Ludność względem 1975")
#     ax_pop.set_ylabel("Liczba ludności (1975 = 100)")
#     ax_pop.set_xlabel("Rok")
    
#     ax_bu.set_title("Powierzchnia terenów zabudowanych względem 1975")
#     ax_bu.set_ylabel("Powierzchnia terenów zabudowanych (1975 = 100)")
#     ax_bu.set_xlabel("Rok")
    
#     # Legend only on the right subplot
#     ax_bu.legend().set_visible(False)
#     # ax_pop.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#     plt.tight_layout()
#     fig.savefig(datadir + os.sep + 'zmiana_BU_POP_1975_2020_relative_lineplots_%s.png'%version,dpi=300,bbox_inches='tight')
#     plt.show()

if plot_charts_PL:
    country = 'PL'

    # --- Load data ---
    result_df = gp.read_file(
        datadir + os.sep + f'LAU_{country}_GHSL_54009_{version}.gpkg',
        ignore_geometry=True
    )

    # --- Aggregate data by landscape type (LAU_NAME) ---
    agg_dict = {f'pop_{year}': 'sum' for year in range(1975, 2025, 5)}
    agg_dict.update({f'bu_{year}': 'sum' for year in range(1975, 2025, 5)})
    agg_dict['pow_km2'] = 'sum'

    agg_df = result_df.groupby('LAU_NAME').agg(agg_dict).reset_index()

    # --- Calculate percentage changes ---
    agg_df['pop_change_1975_2020_pct'] = (
        (agg_df['pop_2020'] - agg_df['pop_1975']) / agg_df['pop_1975'] * 100
    )
    agg_df['bu_change_1975_2020_pct'] = (
        (agg_df['bu_2020'] - agg_df['bu_1975']) / agg_df['bu_1975'] * 100
    )

    # --- Population density (2020) ---
    agg_df['pop_dens_2020_pp_km2'] = (agg_df['pop_2020'] / agg_df['pow_km2']).round(0).astype(int)

    # --- Prepare summary table (Table 4.1) ---
    table_4_1 = agg_df[[
        'LAU_NAME', 'pop_1975', 'pop_2020', 'pop_change_1975_2020_pct',
        'bu_1975', 'bu_2020', 'bu_change_1975_2020_pct', 'pop_dens_2020_pp_km2'
    ]].copy()

    # Convert built-up area from m² to km²
    table_4_1['bu_1975'] = table_4_1['bu_1975'] / 1000
    table_4_1['bu_2020'] = table_4_1['bu_2020'] / 1000
    table_4_1 = table_4_1.rename(columns={"bu_1975": "bu_1975_km2", "bu_2020": "bu_2020_km2"})

    # Round and export
    table_4_1 = table_4_1.round({
        'pop_1975': 0,
        'pop_2020': 0,
        'pop_change_1975_2020_pct': 2,
        'bu_1975_km2': 2,
        'bu_2020_km2': 2,
        'bu_change_1975_2020_pct': 2,
        'pop_dens_2020_pp_km2': 2
    })
    table_4_1.to_csv(datadir + os.sep + f'BU_POP_change_1975-2020_{version}.csv', index=False)
    table_4_1.to_clipboard(index=False)
    print(table_4_1)

    # --- Bar chart: population vs built-up change ---
    plot_df = agg_df.sort_values(by='pop_change_1975_2020_pct', ascending=False)

    ymin = min(plot_df['pop_change_1975_2020_pct'].min(), plot_df['bu_change_1975_2020_pct'].min())
    ymax = max(plot_df['pop_change_1975_2020_pct'].max(), plot_df['bu_change_1975_2020_pct'].max())
    ymin -= (ymax - ymin) * 0.01
    ymax += (ymax - ymin) * 0.1

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(plot_df))

    pop_bars = ax1.bar(index, plot_df['pop_change_1975_2020_pct'], bar_width,
                       label='Population change 1975–2020 [%]', color='tab:blue')

    ax2 = ax1.twinx()
    bu_bars = ax2.bar([i + bar_width for i in index],
                      plot_df['bu_change_1975_2020_pct'], bar_width,
                      label='Built-up area change 1975–2020 [%]', color='tab:orange')

    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels([])

    ax1.set_ylabel('Population change [%]')
    ax2.set_ylabel('Built-up area change [%]')
    plt.title('Change in built-up area and population (1975–2020) by landscape type')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(datadir + os.sep + f'change_BU_POP_1975_2020_bars_{version}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Time series boxplots for population and built-up area ---
    pop_cols = [f"pop_{year}" for year in range(1975, 2025, 5)]
    bu_cols = [f"bu_{year}" for year in range(1975, 2025, 5)]
    years = [int(col.split("_")[1]) for col in pop_cols]

    # Compute relative values (1975 = 100)
    pop_rel = agg_df[pop_cols].div(agg_df['pop_1975'], axis=0) * 100
    bu_rel = agg_df[bu_cols].div(agg_df['bu_1975'], axis=0) * 100

    # Create figure with two panels (side-by-side)
    fig, (ax_bu,ax_pop ) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, dpi=300)

    # --- Population ---
    ax_pop.boxplot(pop_rel.values, positions=range(len(years)), widths=0.6)
    ax_pop.set_xticks(range(len(years)))
    ax_pop.set_xticklabels(years)
    ax_pop.set_title("Population relative to 1975")
    ax_pop.set_ylabel("Population (1975 = 100)")
    ax_pop.set_xlabel("Year")

    # --- Built-up area ---
    ax_bu.boxplot(bu_rel.values, positions=range(len(years)), widths=0.6)
    ax_bu.set_xticks(range(len(years)))
    ax_bu.set_xticklabels(years)
    ax_bu.set_title("Built-up surface area relative to 1975")
    ax_bu.set_ylabel("Built-up surface area (1975 = 100)")
    ax_bu.set_xlabel("Year")

    plt.tight_layout()
    fig.savefig(datadir + os.sep + f'change_BU_POP_1975_2020_boxplots_{version}.png', dpi=300, bbox_inches='tight')
    plt.show()
 
    
    
if generate_tables_PL:
    country='PL'
    # Load the final databse as a df
    result_df = gp.read_file(datadir+os.sep+'LAU_%s_GHSL_54009_%s.gpkg'%(country,version),ignore_geometry=True)

    # Find all pop_ and bu_ columns
    pop_cols = [c for c in result_df.columns if c.startswith("pop_")]
    bu_cols = [c for c in result_df.columns if c.startswith("bu_")]
    
    # Define the aggregation dict
    aggfunc = {col: "sum" for col in pop_cols + bu_cols}
    
    # Add surface function to the dict
    aggfunc["pow_km2"] = "sum"
    
    # Group by LAU name
    shp_grouped = result_df.groupby("LAU_NAME", as_index=False).agg(aggfunc)
    shp_grouped["pow_km2"] = shp_grouped["pow_km2"].round(2)
    
    # Create population summary
    pop_table = shp_grouped[['LAU_NAME', 'pow_km2'] + pop_cols].copy()
    
    # Round all population columns to 0 decimals (whole numbers)
    pop_table[pop_cols] = pop_table[pop_cols].round(0).astype(int)

    print(pop_table)
    pop_table.to_csv(datadir + os.sep + 'POP_1975-2020_%s.csv'%version, index=False)
    
    # Create built-up area summary
    bu_table = shp_grouped[['LAU_NAME', 'pow_km2'] + bu_cols].copy()
    
    # Round all bu columns to km2
    bu_table[bu_cols] = bu_table[bu_cols]/1000000
    bu_table[bu_cols] = bu_table[bu_cols].round(2)
    col_dict = dict(zip(bu_cols, [b+'_km2' for b in bu_cols]))
    bu_table = bu_table.rename(columns=col_dict)

    print(bu_table)
    bu_table.to_csv(datadir + os.sep + 'BU_1975-2020_%s.csv'%version, index=False)
            
# Calulate pop and bu for each class of settlements
if calc_class_stats:
    for year in years:
        raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        raster_pop = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        with rasterio.open(raster_bu) as src: nd=src.nodata
        
        shp = gp.read_file(LAU_gp)
        shp['landid_num']=np.arange(1,len(shp)+1)    
        shp[['xmin','ymin','xmax','ymax']]=shp.bounds
        total_overall=len(shp)
        counter_overall=0
        for country,countrydf in shp.groupby('CNTR_CODE'):           
            processdf = countrydf  
            total=len(processdf)   
            lsmdata=[]
            counter=0

            # Load the pre-computed tertiles for bu and pop arrays
            tertiles_path = os.path.join(datadir, 'tertiles_%s_2020_%s.csv'%(country, version))
            tertiles_d = {}
            with open(tertiles_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['key']
                    value1 = float(row['value1'])
                    value2 = float(row['value2'])
                    tertiles_d[key] = np.array([value1, value2])
                    
            for i,row in processdf.iterrows():
                counter+=1
                counter_overall+=1
                landid = row.landid_num
                aname = row.LAU_NAME
                
                try:
                    buarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_bu)  
                    poparr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_pop)  
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],datadir+os.sep+'ref-lau-2011-01m_%s.tif'%version) 
                except:
                    print('outside of domain')
                    continue
                    # catch error if lau areas are outside the raster data domain (eg overseas territories)
                
                try:
                    # Create strata raster. First, initialize empty outputs
                    strata_class = {}
                    for stuple in [('BUILT_S', buarr), ('POP', poparr)]:
                        terts = tertiles_d[stuple[0]]
                        # Create empty array of the same size as input data
                        result = np.full(buarr.shape, np.nan)
                        # Assign deensity classes
                        result[stuple[1] <= terts[0]] = 0
                        result[(stuple[1] > terts[0]) & (stuple[1] <= terts[1])] = 1
                        result[stuple[1] > terts[1]] = 2
                        # Mask out nodata values
                        strata_class[stuple[0]] = np.where(stuple[1]==nd, np.nan, result)  
                  
                    pop_class = strata_class['POP']
                    bu_class = strata_class['BUILT_S']
                    
                    strata = np.full(bu_class.shape, np.nan)
                    # Where both are valid (i.e., not NaN), assign combined class
                    valid_mask = (~np.isnan(pop_class)) & (~np.isnan(bu_class))
                    strata[valid_mask] = 1 + 3 * pop_class[valid_mask] + bu_class[valid_mask] 
                    
                except:
                    print('here stratification failed')
                    continue
                                   
                try:
                    curr_bu = buarr.copy()
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if lau areas are outside the raster data domain (eg overseas territories)  

                # Generate LAU metrics for each settlement class present in the stratified LAU
                unique_strata = np.unique(strata)
                unique_strata = unique_strata[~np.isnan(unique_strata)]
                for aclass in unique_strata:
                    print(aclass)
                    class_bin = np.where(strata==aclass,1,0)
                    class_bin_masked = class_bin.astype(float).copy()
                    class_bin_masked[landarr!=landid]=np.nan 
                    
                    buarr_strata = np.where(class_bin_masked>0, buarr, np.nan)
                    poparr_strata = np.where(class_bin_masked>0, poparr, np.nan)
                    
                    
                    if np.nansum(class_bin_masked)==0:
                        print('NBU LAU?')            
                        continue
                    
                    label_arr = scipy.ndimage.label(class_bin_masked)[0]
                    segdf = pd.DataFrame()
                    segdf['segid']=label_arr.flatten()
                    segdf['landid']=landarr.flatten()
                    segdf['bu']=buarr_strata.flatten()
                    segdf['popcount']=poparr_strata.flatten()
                    
                    total_bu = segdf.bu.sum()
                    total_pop = segdf.popcount.sum()                    
                                
                    lsmdata.append([landid,aname, aclass,total_bu,total_pop])
                
                print(year,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                
            lsmdatadf=pd.DataFrame(lsmdata) 
            lsmdatadf.columns=['landid','LAU_NAME','strata_class','total_bu','total_pop']
            
            lsmdatadf.to_csv(datadir+os.sep+'LAU_strata_stats_%s_%s_%s.csv' %(country,year, version),index=False) 
             
    # Combine all LAU metrics for all year for each country into one dataframe    
    for country,countrydf in shp.groupby('CNTR_CODE'):   
        alldf = pd.DataFrame()    
        for year in years:       
            try:
                countrydf = pd.read_csv(datadir+os.sep+'LAU_strata_stats_%s_%s_%s.csv' %(country,year, version))
                countrydf['country'] = country
                countrydf['year'] = year
    
            except:
                continue
            alldf = pd.concat([alldf,countrydf], ignore_index=True)

        # Save
        alldf.to_csv(datadir+os.sep+'LAU_class_stats_all_%s_%s.csv' %(country,version),index=False)          
            
# Calulate LAU metrics for each class of settlements
if calc_lsm:
    for year in years:
        raster_bu = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_BUILT_S_GLOBE_R2023A\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_BUILT_S_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        raster_pop = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_POP_GLOBE_R2023A\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0\GHS_POP_Exxxx_GLOBE_R2023A_54009_100_V1_0.tif'.replace('xxxx',str(year))
        with rasterio.open(raster_bu) as src: nd=src.nodata
        
        shp = gp.read_file(LAU_gp)
        shp['landid_num']=np.arange(1,len(shp)+1)    
        shp[['xmin','ymin','xmax','ymax']]=shp.bounds
        total_overall=len(shp)
        counter_overall=0
        for country,countrydf in shp.groupby('CNTR_CODE'):
            
            processdf = countrydf.dissolve(by="LAU_NAME")
            
            processdf = countrydf 
            total=len(processdf)   
            lsmdata=[]
            counter=0

            # Load the pre-computed tertiles for bu and pop arrays
            tertiles_path = os.path.join(datadir, 'tertiles_%s_2020_%s.csv'%(country, version))
            tertiles_d = {}
            with open(tertiles_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = row['key']
                    value1 = float(row['value1'])
                    value2 = float(row['value2'])
                    tertiles_d[key] = np.array([value1, value2])
                    
            for i,row in processdf.iterrows():
                counter+=1
                counter_overall+=1
                landid = row.landid_num
                aname = row.LAU_NAME
                
                try:
                    buarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_bu)  
                    # volarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_vol)  
                    poparr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_pop)  
                    landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],datadir+os.sep+'ref-lau-2011-01m_%s.tif'%version)
                except:
                    print('outside of domain')
                    continue
                    # catch error if lau areas are outside the raster data domain (eg overseas territories)
                
                try:
                    # Create strata raster. First, initialize empty outputs
                    strata_class = {}
                    for stuple in [('BUILT_S', buarr), ('POP', poparr)]:
                        terts = tertiles_d[stuple[0]]
                        # Create empty array of the same size as input data
                        result = np.full(buarr.shape, np.nan)
                        # Assign deensity classes
                        result[stuple[1] <= terts[0]] = 0
                        result[(stuple[1] > terts[0]) & (stuple[1] <= terts[1])] = 1
                        result[stuple[1] > terts[1]] = 2
                        # Mask out nodata values
                        strata_class[stuple[0]] = np.where(stuple[1]==nd, np.nan, result)  
                  
                    pop_class = strata_class['POP']
                    bu_class = strata_class['BUILT_S']
                    
                    strata = np.full(bu_class.shape, np.nan)
                    # Where both are valid (i.e., not NaN), assign combined class
                    valid_mask = (~np.isnan(pop_class)) & (~np.isnan(bu_class))
                    strata[valid_mask] = 1 + 3 * pop_class[valid_mask] + bu_class[valid_mask] 
                    
                except:
                    print('stratification failed')
                    continue
                                   
                try:
                    curr_bu = buarr.copy()
                except:   
                    print('outside of domain')                               
                    continue
                    # catch error if lau areas are outside the raster data domain (eg overseas territories)  

                # Generate LAU metrics for each settlement class present in the stratified LAU
                unique_strata = np.unique(strata)
                unique_strata = unique_strata[~np.isnan(unique_strata)]
                for aclass in unique_strata:
                    class_bin = np.where(strata==aclass,1,0)
                    class_bin_masked = class_bin.astype(float).copy()
                    class_bin_masked[landarr!=landid]=np.nan 
                    
                    buarr_strata = np.where(class_bin_masked>0, buarr, np.nan)
                    poparr_strata = np.where(class_bin_masked>0, poparr, np.nan)
                    # volarr_strata = np.where(class_bin_masked>0, volarr, np.nan)
                    # plt.imshow(class_bin_masked)
                    # plt.show()         
                    # curr_bu_bb_masked[lauarr!=landid]=0 
                    # print('sum:',np.nansum(class_bin_masked))
                    if np.nansum(class_bin_masked)==0:
                        print('NBU LAU?')            
                        continue
                    # calc lsms here and store in list of lists
                    landsc = pylandstats.Landscape(landscape=class_bin_masked+1,nodata=np.nan,res=(100,100))        
                    try:
                        lpi = landsc.largest_patch_index(class_val=2)
                        number_of_patches = landsc.number_of_patches(class_val=2)
                        landscape_shape_index = landsc.landscape_shape_index(class_val=2)
                        fractal_dimension = landsc.fractal_dimension(class_val=2).mean()
                        perimeter_area_ratio = landsc.perimeter_area_ratio(class_val=2).mean()
                        contagion = landsc.contagion()
                        classsurf = np.nansum(class_bin_masked)
                    except:
                        continue
                    # lpi weighted by total BU
                    label_arr = scipy.ndimage.label(class_bin_masked)[0]
                    segdf = pd.DataFrame()
                    segdf['segid']=label_arr.flatten()
                    segdf['landid']=landarr.flatten()
                    segdf['bu']=buarr_strata.flatten()
                    # segdf['vol']=volarr_strata.flatten()
                    segdf['popcount']=poparr_strata.flatten()
                    segdf=segdf[segdf.segid>0]
                   
                    uq_segments,seg_sizes = np.unique(label_arr,return_counts=True)
                    segsizedf = pd.DataFrame()
                    segsizedf['segid']=uq_segments
                    segsizedf['seg_size']=seg_sizes 
                    segsizedf=segsizedf[segsizedf.segid>0]
                    segsizedf = segsizedf.sort_values(by='seg_size',ascending=False)
                    largest_segid=segsizedf.segid.values[0]
                    largest_seg_size=segsizedf.seg_size.values[0]        
                    largest_seg_bu = segdf[segdf.segid==largest_segid].bu.sum()
                    # largest_seg_vol = segdf[segdf.segid==largest_segid].vol.sum()
                    largest_seg_pop = segdf[segdf.segid==largest_segid].popcount.sum()
                    pop_largest_5 = segdf[segdf.segid.isin(segsizedf.head(5).segid.values)].popcount.sum()
                    pop_largest_10 = segdf[segdf.segid.isin(segsizedf.head(10).segid.values)].popcount.sum()
            
                    total_bu = segdf.bu.sum()
                    # total_vol = segdf.vol.sum()
                    total_pop = segdf.popcount.sum()        
                    lpi_weighted_bu = largest_seg_bu / total_bu
                    # lpi_weighted_vol = largest_seg_vol / total_vol
                    lpi_weighted_pop = largest_seg_pop / total_pop
                    
                    segment_busums = segdf.groupby('segid').bu.sum()
                    # segment_volsums = segdf.groupby('segid').vol.sum()
                    segment_popsums = segdf.groupby('segid').popcount.sum()
                   
                    segment_busums_min = np.min(segment_busums)
                    segment_busums_max = np.max(segment_busums)
                    segment_busums_mean = np.mean(segment_busums)
                    segment_busums_med = np.median(segment_busums)
            
                    # segment_volsums_min = np.min(segment_volsums)
                    # segment_volsums_max = np.max(segment_volsums)
                    # segment_volsums_mean = np.mean(segment_volsums)
                    # segment_volsums_med = np.median(segment_volsums)
            
                    segment_popsums_min = np.min(segment_popsums)
                    segment_popsums_max = np.max(segment_popsums)
                    segment_popsums_mean = np.mean(segment_popsums)
                    segment_popsums_med = np.median(segment_popsums)
                    
                    segments_below_popcount10 = segment_busums[segment_popsums<10].shape[0]
                    segments_below_popcount50 = segment_busums[segment_popsums<50].shape[0]
                    segments_below_popcount100 = segment_busums[segment_popsums<100].shape[0]
                    segments_total = len(segment_busums)
                    
                    isolated_cells = len(segsizedf[segsizedf.seg_size==1])
            
                    lsmdata.append([landid,aname,aclass,total_bu,
                                    total_pop,lpi,lpi_weighted_bu,
                                    lpi_weighted_pop,number_of_patches,
                                    landscape_shape_index,fractal_dimension,perimeter_area_ratio,contagion,largest_seg_size,largest_seg_pop,pop_largest_5,pop_largest_10,isolated_cells,
                                    segments_below_popcount10,segments_below_popcount50,segments_below_popcount100,segments_total,
                                    segment_busums_min,segment_busums_max,segment_busums_mean,segment_busums_med,
                                    segment_popsums_min,segment_popsums_max,segment_popsums_mean,segment_popsums_med])
                    # total_vol, lpi_weighted_vol,segment_volsums_min,segment_volsums_max,segment_volsums_mean,segment_volsums_med,
                
                print(year,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                
            lsmdatadf=pd.DataFrame(lsmdata) 
            lsmdatadf.columns=['landid','LAU_NAME','strata_class','total_bu',
                               # 'total_vol',
                               'total_pop','lpi','lpi_weighted_bu',
                               # 'lpi_weighted_vol',
                               'lpi_weighted_pop','number_of_patches',
                               'landscape_shape_index','fractal_dimension','perimeter_area_ratio','contagion','largest_seg_size','largest_seg_pop','pop_largest_5',
                               'pop_largest_10','isolated_cells','segments_below_popcount10','segments_below_popcount50','segments_below_popcount100','segments_total',
                               'segment_busums_min','segment_busums_max','segment_busums_mean','segment_busums_med',
                               # 'segment_volsums_min','segment_volsums_max','segment_volsums_mean','segment_volsums_med',
                               'segment_popsums_min','segment_popsums_max','segment_popsums_mean','segment_popsums_med']
            
            lsmdatadf.to_csv(datadir+os.sep+'LAU_lsm_%s_%s_%s.csv' %(country,year, version),index=False) 
  
    # Combine all LAU metrics for all year for each country into one dataframe    
    for country,countrydf in shp.groupby('CNTR_CODE'):   
        alldf = pd.DataFrame()    
        for year in years:       
            try:
                countrydf = pd.read_csv(datadir+os.sep+'LAU_lsm_%s_%s_%s.csv' %(country,year, version))
                countrydf['country'] = country
                countrydf['year'] = year
    
            except:
                continue
            alldf = pd.concat([alldf,countrydf], ignore_index=True)
        # Add a new metric
        alldf['numpatch_per_capita'] = alldf.number_of_patches / alldf.total_pop
        # Save
        alldf.to_csv(datadir+os.sep+'LAU_lsm_all_%s_%s.csv' %(country,version),index=False) 


if panel_plot_pop_PL:
    country='PL'
    shp = gp.read_file(LAU_gp)
    for country,countrydf in shp.groupby('CNTR_CODE'): 
            
        # Load the final databse as a df
        result_df = pd.read_csv(datadir+os.sep+'LAU_class_stats_all_%s_%s.csv' %(country,version))
           
        # Przygotuj figure i osie
        fig, axes = plt.subplots(3, 3, figsize=(14, 12),
                                 sharex=True, sharey=False, dpi=300)
        
        # Mapping strata -> subplot position
        pos_map = {
            1: (2, 0), 2: (2, 1), 3: (2, 2),
            4: (1, 0), 5: (1, 1), 6: (1, 2),
            7: (0, 0), 8: (0, 1), 9: (0, 2)
        }
        
        for strata, label in strata_names.items():
            r, c = pos_map[strata]
            ax = axes[r, c]
        
            # Filtrowanie danych dla danej straty
            subdf = result_df[result_df["strata_class"] == strata]
        
            # Rysuj osobną linię dla każdego typu krajobrazu
            for landid, group in subdf.groupby("landid"):
                ax.plot(group["year"], group["total_pop"],
                        label=landid, linestyle="-")
        
            # Tytuły i siatka
            ax.set_title(label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        
        # Wspólne etykiety osi
        fig.text(0.5, 0.0, "Rok", ha="center", fontsize=11)
        fig.text(0.0, 0.5, "Liczba ludności", va="center", rotation="vertical")
        
        # # Legenda tylko raz, na zewnątrz
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc="center right",
        #            bbox_to_anchor=(1.1, 0.5), title="Typ krajobrazu")
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        fig.savefig(datadir + os.sep + 'POP_1975-2020_class_multipanel_%s.png'%version,dpi=300,bbox_inches='tight')
        plt.show()
               
        ## Wartoci względem 1975 (=100)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12),
                             sharex=True, sharey=False, dpi=300)
        
        for strata, label in strata_names.items():
            r, c = pos_map[strata]
            ax = axes[r, c]
        
            # Dane dla tej straty
            subdf = result_df[result_df["strata_class"] == strata]
        
            # Linie dla każdego krajobrazu
            for landid, group in subdf.groupby("landid"):
                group = group.sort_values("year")  # porządek chronologiczny
                base_val = group.loc[group["year"] == 1975, "total_pop"].values
                if len(base_val) == 0 or base_val[0] == 0:  
                    continue  # pomijamy brak danych lub 0
                norm_vals = group["total_pop"] / base_val[0] * 100
        
                ax.plot(group["year"], norm_vals, label=landid, linestyle="-")
        
            ax.set_title(label, fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        
        # Opisy osi
        fig.text(0.5, 0.0, "Rok", ha="center", fontsize=11)
        fig.text(0.0, 0.5, "Liczba ludności (1975 = 100)", va="center", rotation="vertical")
        
        # # Legenda tylko raz
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc="center right",
        #            bbox_to_anchor=(1.1, 0.5), title="Typ krajobrazu")
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        fig.savefig(datadir + os.sep + 'POP_1975-2020_class_relative_multipanel_%s.png'%version,dpi=300,bbox_inches='tight')
        plt.show()

if series_plot_lsm_PL:
    country='PL'
    # Load the final databse as a df
    alldf = pd.read_csv(datadir+os.sep+'LAU_lsm_all_%s_%s.csv' %(country,version))
    # For ech landscape metric, plot timeseries of builtup-related metric values in LAU regions
    bupop = ['total_bu','total_pop']
    lpi = ['lpi']
    lsm = ['number_of_patches', 'contagion','fractal_dimension', 'perimeter_area_ratio',]
    # do not show redundant / not visible metrics: 'isolated_cells','segments_total', 'landscape_shape_index', 'largest_seg_size'
    
    lpi_GHSL = ['lpi_weighted_bu', 'lpi_weighted_pop']
    patch_GHSL = ['largest_seg_pop',  'segments_below_popcount100',
                  'segment_busums_med', 'segment_popsums_med',]
    # 'segments_below_popcount10','segments_below_popcount50',
    # 'pop_largest_5','pop_largest_10',
    #'segment_busums_min','segment_busums_max', 'segment_busums_mean', 
    # 'segment_popsums_min',  'segment_popsums_max', 'segment_popsums_mean', 
    
    # Seaborn style
    sns.set(style="whitegrid")
    
    # Create two figurs: one for landscape metrics, second on for GHSL-induced metrics
    for plotcols in [bupop, lpi, lsm, lpi_GHSL,patch_GHSL ]:
        # Set up subplot grid
        ncols = 2
        nrows = (len(plotcols) + 2) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4 * nrows),
                                 sharex=True, dpi=150)
        axes = axes.flatten()
        
        # Define color palette with distinct colors
        palette = sns.color_palette("Set3")  # Try "Paired", "colorblind", "Set3" as alternatives
        
        # Loop through columns
        for i, col in enumerate(plotcols):
            ax = axes[i]
            sns.boxplot(
                data=alldf,
                x='strata_class',
                y=col,
                hue='year',
                ax=ax,
                # palette=palette,
                showfliers=False,
                linewidth=0.5
            )
            # Set x-axis labels from strata_names
            ax.set_xticks([x-1 for x in strata_names.keys()])
            ax.set_xticklabels(strata_names.values(), rotation=45, ha='right')
        
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel("Settlement Class")
            ax.set_ylabel(col)
        
            # Hide all legends except the last one
            if i < len(plotcols):
                ax.get_legend().remove()
        
        # Remove unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            legend_ax = axes[j]  # Use this one to place the legend
        
        plt.tight_layout()
        plt.show()
    
panel_plot_lsm_PL_norm=True
if panel_plot_lsm_PL_norm:
    country = 'PL'

    # Load the main database
    lsm_df = pd.read_csv(datadir + os.sep + f'LAU_lsm_all_{country}_{version}.csv')
    
    # Add metric DI from https://www.tandfonline.com/doi/full/10.1080/17538947.2018.1474957#d1e484
    NP = lsm_df.number_of_patches
    LP = lsm_df.lpi
    lsm_df['NPn'] = (NP - NP.min()) / (NP.max() - NP.min()) * 100
    lsm_df['LPn'] = (LP - LP.min()) / (LP.max() - LP.min()) * 100
    lsm_df['DI'] = (lsm_df.NPn + (100-lsm_df.LPn))/2
    # Prepare figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(10,10),
                             sharex=True, sharey=True, dpi=150)
    ametric = 'number_of_patches'
    # Mapping of strata to subplot positions
    pos_map = {
        1: (2, 0), 2: (2, 1), 3: (2, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (0, 0), 8: (0, 1), 9: (0, 2)
    }

    for strata, label in strata_names.items():
        r, c = pos_map[strata]
        ax = axes[r, c]
    
        # Filter data for the given strata
        subdf = lsm_df[lsm_df["strata_class"] == strata]
    
        # Calculate yearly means
        yearly_stats = subdf.groupby("year")[ametric].agg(['mean', 'median']).reset_index()
        bu_stats = subdf.groupby("year")["total_bu"].mean().reset_index()
        pop_stats = subdf.groupby("year")["total_pop"].mean().reset_index()
    
        # # Normalize all to 1975 = 100%
        # for df in [yearly_stats, bu_stats, pop_stats]:
        #     for col in df.columns[1:]:
        #         base = df.loc[df["year"] == 1975, col].values[0]
        #         df[col] = df[col] / base * 100
                
        # OR calculate z-score!
        for df in [yearly_stats, bu_stats, pop_stats]:
            for col in df.columns[1:]:
                df[col] = scipy.stats.zscore(df[col])
                
        # Plot mean and median of the main metric
        ax.plot(
            yearly_stats["year"], yearly_stats["mean"],
            color="steelblue", linewidth=2, marker="o", markersize=5,
            alpha=0.9, label=f"Mean {ametric} (norm.)", markevery=2
        )

        # Built-up area line (orange, dashed)
        ax.plot(
            bu_stats["year"], bu_stats["total_bu"],
            color="darkorange", linestyle="--", linewidth=2,
            marker="s", markersize=5, alpha=0.9,
            label="Built-up surface area (norm.)", markevery=2
        )

        # Population line (crimson, dotted)
        ax.plot(
            pop_stats["year"], pop_stats["total_pop"],
            color="crimson", linestyle=":", linewidth=2,
            marker="D", markersize=5, alpha=0.9,
            label="Population (norm.)", markevery=2
        )
    
        # Titles and grid
        ax.set_title(label, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Label for right y-axis
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        # ax2.set_ylabel("Built-up / Population", color="gray", fontsize=9)
        # Add line at zero
        ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)


    # Shared axis labels
    # fig.text(0.5, 0.0, "Year", ha="center", fontsize=11)
    # fig.text(0.0, 0.5, ametric.replace("_", " ").title(), va="center", rotation="vertical")
    fig.text(0.0, 0.5, 'z-score', va="center", rotation="vertical")

    # Handle legends
    handles1, labels1 = ax.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc="lower center", fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(datadir + os.sep + f"{ametric}_zscore_1975-2020_multipanel_{version}.png", dpi=300, bbox_inches='tight')
    plt.show()

    
    
# if panel_plot_lsm_PL:
#     country='PL'
#     #ametric = 'numpatch_per_capita'
#     ametric = 'number_of_patches'
#     # Load the final databse as a df
#     lsm_df = pd.read_csv(datadir+os.sep+'LAU_lsm_all_%s_%s.csv' %(country,version))
            
#     # Przygotuj figure i osie
#     fig, axes = plt.subplots(3, 3, figsize=(14, 12),
#                              sharex=True, sharey=False, dpi=300)
    
#     # Mapping strata -> subplot position
#     pos_map = {
#         1: (2, 0), 2: (2, 1), 3: (2, 2),
#         4: (1, 0), 5: (1, 1), 6: (1, 2),
#         7: (0, 0), 8: (0, 1), 9: (0, 2)
#     }
    
#     for strata, label in strata_names.items():
#         r, c = pos_map[strata]
#         ax = axes[r, c]
    
#         # Filtrowanie danych dla danej straty
#         subdf = lsm_df[lsm_df["strata_class"] == strata]
    
#         # Rysuj osobną linię dla każdego typu krajobrazu
#         for landid, group in subdf.groupby("landid"):
#             ax.plot(group["year"], group[ametric],
#                     label=landid, linestyle="-")
    
#         # Tytuły i siatka
#         ax.set_title(label, fontsize=11)
#         ax.grid(True, linestyle="--", alpha=0.5)
    
#     # Wspólne etykiety osi
#     fig.text(0.5, 0.0, "Rok", ha="center", fontsize=11)
#     fig.text(0.0, 0.5, ametric, va="center", rotation="vertical")

    
#     plt.tight_layout(rect=[0, 0, 0.9, 1])
#     fig.savefig(datadir + os.sep + '%s_1975-2020_multipanel_%s.png'%(ametric,version),dpi=300,bbox_inches='tight')
#     plt.show()
    
    
    
    # ## RELATIVE values
    # fig, axes = plt.subplots(3, 3, figsize=(14, 12),
    #                      sharex=True, sharey=True, dpi=300)
    
    # for strata, label in strata_names.items():
    #     r, c = pos_map[strata]
    #     ax = axes[r, c]
    
    #     # Dane dla tej straty
    #     subdf = lsm_df[lsm_df["strata_class"] == strata]
    
    #     # Linie dla każdego krajobrazu
    #     for landid, group in subdf.groupby("landid"):
    #         group = group.sort_values("year")  # porządek chronologiczny
    #         base_val = group.loc[group["year"] == 1975, ametric].values
    #         if len(base_val) == 0 or base_val[0] == 0:  
    #             continue  # pomijamy brak danych lub 0
    #         norm_vals = group[ametric] / base_val[0] * 100
    #         if norm_vals.iloc[0]!=100: print(landid)
    
    #         ax.plot(group["year"], norm_vals, label=landid, linestyle="-")
    
    #     ax.set_title(label, fontsize=11)
    #     ax.grid(True, linestyle="--", alpha=0.5)
    
    # # Opisy osi
    # fig.text(0.5, 0.0, "Rok", ha="center", fontsize=11)
    # fig.text(0.0, 0.5, "%s (1975 = 100)"%ametric, va="center", rotation="vertical")
    
    # # # Legenda tylko raz
    # # handles, labels = ax.get_legend_handles_labels()
    # # fig.legend(handles, labels, loc="center right",
    # #            bbox_to_anchor=(1.1, 0.5), title="Typ krajobrazu")
    
    # plt.tight_layout(rect=[0, 0, 0.9, 1])
    # fig.savefig(datadir + os.sep + '%s_1975-2020_relative_multipanel_%s.png'%(ametric,version),dpi=300,bbox_inches='tight')
    # plt.show()

     
if generate_class_table:
    # Load the final databse as a df
    class_df = pd.read_csv(datadir+os.sep+'LAU_class_stats_all_%s_%s.csv' %(country,version))

    # aggregate population by strata and year
    pop_summary = (
        class_df.groupby(["strata_class", "year"])["total_pop"]
        .sum()
        .unstack()  # years as columns
        .reset_index()
    )
    
    # keep only 1975 and 2020
    pop_summary = pop_summary[[ "strata_class", 1975, 2020 ]]
    
    # rename columns
    pop_summary = pop_summary.rename(
        columns={1975: "Ludność 1975", 2020: "Ludność 2020"}
    )
    
    # Round pop counts
    pop_summary["Ludność 1975"] =pop_summary["Ludność 1975"].round(0)
    pop_summary["Ludność 2020"]=pop_summary["Ludność 2020"].round(0)
    
    # add class names
    pop_summary["Klasa"] = pop_summary["strata_class"].map(strata_names)
    
    # calculate change
    pop_summary["Zmiana ludności 1975-2020"] = (
        pop_summary["Ludność 2020"] - pop_summary["Ludność 1975"]).round(0)
    
    # calculate % change
    pop_summary["Zmiana ludności 1975-2020 [%]"] = (
        (pop_summary["Ludność 2020"] - pop_summary["Ludność 1975"])
        / pop_summary["Ludność 1975"] * 100
    ).round(2)
    
    # reorder cols
    pop_summary = pop_summary[["Klasa", "Ludność 1975", "Ludność 2020",
                               "Zmiana ludności 1975-2020", "Zmiana ludności 1975-2020 [%]"]]
    
    # Save
    print(pop_summary)
    pop_summary.to_csv(datadir+os.sep+'LAU_class_POP_1975_2020_%s.csv' %version,index=False) 
    pop_summary.to_clipboard()
             
   

if calc_agrm:
    print(' lets calculate agreement!')

    for y, period in enumerate(agr_periods):
        print(period)
        
        for s in ['BUILT_S', 'POP']:
            raster_start = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_%s_GLOBE_R2023A\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0.tif'%(s,s, period[0],s, period[0])
            raster_end = r'C:\DATA\GHSL_PRODUCTS\GHSL_R2023\DATA\GHS_%s_GLOBE_R2023A\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0\GHS_%s_E%s_GLOBE_R2023A_54009_100_V1_0.tif'%(s,s, period[1],s, period[1])
            
            
        
            shp = gp.read_file(LAU_gp,encoding='latin')
            shp['landid_num']=np.arange(1,len(shp)+1)    
            shp[['xmin','ymin','xmax','ymax']]=shp.bounds
            total_overall=len(shp)
            counter_overall=0
            for country,countrydf in shp.groupby('CNTR_CODE'):
                
                processdf = countrydf
                total=len(processdf)   
                agrmdata=[]
                counter=0
                
                for i,row in processdf.iterrows():
                    counter+=1
                    counter_overall+=1
                    aname = row.LAU_NAME
                    
                    try:
                        arr_start = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_start)  
                        arr_end = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],raster_end)  
                        landarr = get_subset([row.xmin,row.ymin,row.xmax,row.ymax],datadir+os.sep+'ref-lau-2011-01m_%s.tif'%version)  
                        
                    except:
                        print('outside of domain')
                        continue
                        # catch error if land areas are outside the raster data domain (eg overseas territories)            
                    
                    landid = row.landid_num
                    try:
                        _start = arr_start.copy()         
                        _end = arr_end.copy()
                    except:   
                        print('outside of domain')                               
                        continue
                        # catch error if land areas are outside the raster data domain (eg overseas territories)  
                        
                    _start_masked = _start.astype(float).copy()
                    _start_masked[landarr!=landid]=np.nan 
                    
                    _end_masked = _end.astype(float).copy()
                    _end_masked[landarr!=landid]=np.nan 
                    # plt.imshow(bu_start_masked)
                    # plt.show()         
                    # curr_bu_bb_masked[landarr!=landid]=0 
                    if np.nansum(_start_masked)==0 or np.nansum(_end_masked)==0:
                        print('NBU land?')            
                        continue
                    # calc agreement measures here and store in list of lists
                    try:
                        # Compute agreement metrics
                        cJaccard = cont_jaccard(_start_masked, _end_masked)
                        cPrecision = cont_precision(_end_masked, _start_masked)
                        cRecall = cont_recall(_end_masked, _start_masked)
                        
                        # Compute difference measures
                        _RMSD = RMSD(_end_masked, _start_masked)
                        _MAD = MAD(_end_masked, _start_masked)
                        _MD = MD(_end_masked, _start_masked)
                        difference = np.nansum(_end_masked)-np.nansum(_start_masked)
                        change_rate = CR(_start_masked, _end_masked)
                        
                    except:
                        continue
            
                    agrmdata.append([landid,aname, cJaccard, cPrecision, cRecall, _RMSD, _MAD, _MD, difference, change_rate])
                    
                    print(period,country,counter,'/',total,counter_overall,'/',total_overall,landid)
                    
                agrmdatadf=pd.DataFrame(agrmdata)
                agrmdatadf.columns=['landid','name','cJaccard','cPrecision','cRecall', '_RMSD', '_MAD', '_MD', 'difference', 'change_rate']
                
                agrmdatadf.to_csv(datadir+os.sep+'LAU_agrm_%s_%s_%s_%s.csv' %(s, country,period[0], period[1]),index=False) 

    print('join agreement measures into one geopackage!')
    for y, period in enumerate(agr_periods):
        print(period)
        for s in ['BUILT_S', 'POP']:

            shp = gp.read_file(LAU_gp,encoding='latin')
            shp.LAU_NAME = shp.LAU_NAME.map(str)
        
            shp['landid_num']=np.arange(1,len(shp)+1)   
            countries=shp.CNTR_CODE.unique()
            alldf = pd.DataFrame()
            for country in countries:
                try:
                    countrydf = pd.read_csv(datadir+os.sep+'LAU_agrm_%s_%s_%s_%s.csv' %(s, country,period[0], period[1]))
                except:
                    continue
                alldf = pd.concat([alldf,countrydf], ignore_index=True)
                print(country)
                
            shp_joined = shp.merge(alldf,left_on='landid_num',right_on='landid',how='left')
            
            # calc the ratios:
            shp_joined['year_start'] = int(period[0])
            shp_joined['year_end'] = int(period[1])
    
            # shp_joined.crs=None  
            del shp['LAU_NAME']
            shp_joined.to_file(datadir+os.sep+'LAU_agrm_%s_results_%s_%s_%s.gpkg' %(s, version, period[0], period[1]),driver='GPKG')


if plot_agr_maps_PL:
    country='PL'
    for period in agr_periods:
        # Load the final database with agr measures
        bu_gdf = gp.read_file(datadir+os.sep+'LAU_agrm_BUILT_S_results_%s_%s_%s.gpkg' %(version, period[0], period[1]))
        pop_gdf = gp.read_file(datadir+os.sep+'LAU_agrm_POP_results_%s_%s_%s.gpkg' %(version, period[0], period[1]))
        # Reproject the gdfs to a Polish projection CRS Polkovo
        bu_gdf = bu_gdf.to_crs(2180)
        pop_gdf = pop_gdf.to_crs(2180)
        # Select metric for plot
        plotcol = 'cJaccard'
        # Classify agreement into agreement categories
        _bins = [0.2,0.3,0.4,0.5,0.6,0.7,0.82]
        # Create labels for the bins
        _labels = ["0.2-0.3","0.3-0.4","0.4-0.5", "0.5-0.6", "0.6-0.7","0.7-0.8"]
        # Assign categories
        bu_gdf["agr_class"] = pd.cut(bu_gdf[plotcol], bins=_bins, labels=_labels, right=False)
        pop_gdf["agr_class"] = pd.cut(pop_gdf[plotcol], bins=_bins, labels=_labels, right=False)
        
        # -------------------------------------------------------
        # Plot Figure 1: Built-up area agreement
        # -------------------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
        
        bu_gdf.plot(column='agr_class',legend=False, ax=ax[0], edgecolor="black", linewidth=0.1, cmap="coolwarm_r",)
        ax[0].set_title("Built-up surface distribution agreement between 1975 and 2020", fontsize=14)
        ax[0].axis("off")
        
        # -------------------------------------------------------
        # Plot Figure 2: Population agreement
        # -------------------------------------------------------
        pop_gdf.plot(column='agr_class', legend=True, ax=ax[1], edgecolor="black", linewidth=0.1, cmap="coolwarm_r", )
        ax[1].set_title("Population distribution agreement between 1975–2020", fontsize=14)
        ax[1].axis("off")
        
        # Set legend title
        leg=ax[1].get_legend()
        leg.set_title("cJaccard [0-1]:")
        
        plt.tight_layout()
        fig.savefig(datadir + os.sep + 'agreement_BU_POP_%s_%s_map_%s.png'%(period[0],period[1],version),dpi=300,bbox_inches='tight')
        plt.show()
  
    
    
    
    