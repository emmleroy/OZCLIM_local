import re
import os 
import xarray as xr 
import pandas as pd
import numpy as np
from typing import Pattern, List
import gcpy.constants as gcon
import glob
import gcgridobj 
import regionmask 
import matplotlib.pyplot as plt 

# DEFINE DIRECTORIES HERE
MODEL_OUTPUT_DIR = "/Users/emmie/Documents/OZCLIM_local/data/"
CASTNET_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/ExtData/CASTNET/"
CNEMC_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/ExtData/CNEMC/"
AIRBASE_DIR = '/home/eleroy/proj-dirs/OZCLIM/data/ExtData/EEA/'

# import example c48 grid
dir = f'{MODEL_OUTPUT_DIR}/tools/cs48_example.nc4'
cs48_example = xr.open_dataset(dir)
dst_grid = gcgridobj.cstools.extract_grid(cs48_example)

# import GCHP variables to skip
skip_vars = gcon.skip_these_vars


def set_matplotlib_font(font_family: str):
    """Set the matplotlib font family.
    Args:
        font_family (str): the font family
    """
    supported_fonts: List[str] = {'Andale Mono', 'Arial', 'Arial Black',
                               'Comic Sans MS', 'Courier New', 'Georgia',
                               'Impact', 'Times New Roman', 'Trebuchet MS',
                               'Verdana', 'Webdings', 'Amiri', 'Lato'}

    assert font_family in supported_fonts, f'Font {font_family} not supported.'
    plt.rcParams['font.family'] = font_family
    plt.rcParams["mathtext.fontset"] = 'stixsans'


def get_file_list(directory_path: str, compiled_regex: Pattern):
    """Return a list of file paths in a directory matching the regex pattern.

    Args:
        directory_path (str): path to directory where to look for files
        compiled_regex (pattern): compiled regex pattern to match

    Returns:
        file_list (list[str]): list of file paths

    """

    file_list = []
    for file_name in os.listdir(directory_path):
        if compiled_regex.match(file_name):
            file_list.append(os.path.join(directory_path, file_name))

    # Important!!! Sort to concatenate chronologically
    file_list.sort()

    return file_list


def open_multifile_ds(file_list: List[str], compat_check=False):
    """Open multiple files (or a single file) as a single dataset.

    WARNING: compatibility checks are overrided for speed-up!

    Args:
        file_list (list(str): list of file paths

    Returns:
        ds (xr.DataSet): a single concatenated xr.Dataset

    """

    v = xr.__version__.split(".")

    
    if int(v[0]) == 0 and int(v[1]) >= 15:
        if compat_check is False:
            ds = xr.open_mfdataset(file_list,
                                drop_variables=skip_vars,
                                combine='nested',
                                concat_dim='time',
                                engine='netcdf4',
                                chunks='auto',
                                parallel=True,
                                data_vars='minimal',
                                coords='minimal',
                                compat='override',
                                autoclose=True)
        else:
            ds = xr.open_mfdataset(file_list,
                                drop_variables=skip_vars,
                                #combine='nested',
                                #concat_dim='time',
                                engine='netcdf4',
                                #chunks='auto',
                                parallel=True,
                                #data_vars='minimal',
                                #coords='minimal',
                                #compat='override',
                                autoclose=True)            
    else:
        ds = xr.open_mfdataset(file_list, drop_variables=skip_vars, autoclose=True)

    return ds


def get_ds(my_simulation: str, variable_type: str):
    """ Function returns dataset for a given simulation for one of 
        three variable types ("Emissions", "SpeciesConc", or "MDA8_O3").

    Args:
        my_simulation(str): example "w10_ref"

        variable_type(str): "Emissions", "SpeciesConc", or "MDA8_O3"
    """

    directory = MODEL_OUTPUT_DIR

    if variable_type=="MDA8_O3":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.{variable_type}")
    elif variable_type[:4]=="Spec":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.SpeciesConc")
    elif variable_type[:4]=="Emis":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.Emissions")
 
    file_path = get_file_list(directory, regex_pattern)
    print(file_path)
    ds = open_multifile_ds(file_path, compat_check=True)
    
    # Chunk these large multi-year datasets into 1-year chunks
    ds = ds.chunk({'time': (366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365)})

    return ds


def get_ensemble_ds(sim, variable_type):
    """For a given variable type ("Emissions", "SpeciesConc", "MDA8_O3"), 
    return a concatenated dataset with all 5 realizations/ensemble members along dimensions 'sim'
    """
    w10 = get_ds(f"w10_{sim}_c48", variable_type)
    w13 = get_ds(f"w13_{sim}_c48", variable_type)
    w14 = get_ds(f"w14_{sim}_c48", variable_type)
    w26 = get_ds(f"w26_{sim}_c48", variable_type)
    w28 = get_ds(f"w28_{sim}_c48", variable_type)

    ds = xr.concat([w10, w13, w14, w26, w28], dim="sim")
    
    return ds 


def _read_and_process_castnet_data(file_path, qa_options, month):
    df = pd.read_csv(file_path)
    valid_df = df[df['QA_CODE'].isin(qa_options)]
    valid_df['DATE_TIME'] = pd.to_datetime(valid_df['DATE_TIME'], infer_datetime_format=True)
    return valid_df[valid_df['DATE_TIME'].dt.month == month]


def _read_and_process_cnemc_data(directory_path, pattern):
    matching_files = glob.glob(f'{directory_path}/{pattern}')
    combined_df = pd.concat([pd.read_csv(file) for file in matching_files], ignore_index=True)
    o3_df = combined_df[combined_df['type'] == 'O3']
    melted_df = pd.melt(o3_df, id_vars=['date', 'hour', 'type'], var_name='SITE_ID', value_name='OZONE')
    melted_df['OZONE'] = melted_df['OZONE'] * 24.45 / 48  # Convert ug/m3 to ppbv
    # Add a 'DATE_TIME' column
    melted_df['DATE_TIME'] = pd.to_datetime(melted_df['date'].astype(str) + melted_df['hour'].astype(str).str.zfill(2), format='%Y%m%d%H')
    return melted_df[~melted_df['OZONE'].isna()]


def _read_and_process_eea_data(file_path, qa_options, month):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df = df[(df['AveragingTime'] == 'hour') & (df['UnitOfMeasurement'] == 'µg/m3')]
    df['OZONE'] = df['Concentration'] * 24.45 / 48
    valid_df = df[df['Validity'].isin(qa_options) & df['Verification'].isin([1])]
    valid_df = valid_df.rename(columns={"AirQualityStationEoICode": "SITE_ID"})
    valid_df['DATE_TIME'] = pd.to_datetime(valid_df['DatetimeBegin'])
    return valid_df[valid_df['DATE_TIME'].dt.month == month]


def _get_ozone_observations(region, month):
    if region == "ENA":
        file_path =  f'{CASTNET_DIR}/ozone_2014.csv'
        df_month_valid = _read_and_process_castnet_data(file_path, [3], month)

    elif region == "EAS":
        directory_path = f'{CNEMC_DIR}'
        pattern = f'china_sites_2014{str(month).zfill(2)}*.csv' if month in [6, 12] else None
        if not pattern:
            raise Exception("Sorry, only December (12) or June (6) accepted for China")
        df_month_valid = _read_and_process_cnemc_data(directory_path, pattern)

    elif region == "WCE":
        file_path = f'{AIRBASE_DIR}/allcountries_O3_2014.csv'
        df_month_valid = _read_and_process_eea_data(file_path, [1, 2, 3], month)

    else:
        raise Exception("Sorry, only ENA, EAS, or WCE regions accepted")
    df_month_valid['DATE'] = pd.to_datetime(df_month_valid['DATE_TIME'], infer_datetime_format=True).dt.date
    return df_month_valid


def _get_all_site_locations(region):
    # Define file paths and column names to rename for each region
    region_files = {
        "ENA": (f"{CASTNET_DIR}/activesuspendedcastnetsites.xlsx", 'Site ID and Webpage Link'),
        "EAS": (f"{CNEMC_DIR}/SiteList.csv", 'Site'),
        "WCE": (f"{AIRBASE_DIR}/DataExtract.csv", 'Air Quality Station EoI Code'),
    }
    
    if region not in region_files:
        raise Exception("Sorry, only ENA, EAS, or WCE regions accepted")
    
    file_path, column_name = region_files[region]
    
    # Load data and rename the specified column to 'SITE_ID'
    if region == "ENA":
        site_locations_df = pd.read_excel(file_path)
    else:
        site_locations_df = pd.read_csv(file_path)
    
    site_locations_df.rename(columns={column_name: 'SITE_ID'}, inplace=True)
    
    # Drop duplicates based on 'SITE_ID'
    site_locations_df.drop_duplicates(subset='SITE_ID', inplace=True)

    return site_locations_df 


def _filter_sites_by_observation_availability(region, month, criteria=90):

    df = _get_ozone_observations(region, month)

    # Define the number of days in each month of interest
    days_in_month_dict = {6: 30, 7: 31, 12: 31}
    
    # Calculate the total possible observations for the given month (days * 24 hours)
    total_possible_observations = days_in_month_dict[month] * 24
    
    # Group by SITE_ID and calculate the percentage of available observations
    df_clean = df[~df.OZONE.isin([np.nan])]
    percentage_obs_per_site = df_clean.groupby('SITE_ID').size() / total_possible_observations * 100
    
    # Identify sites with at least the specified criteria percentage of availability
    eligible_sites = percentage_obs_per_site[percentage_obs_per_site >= criteria].index
    
    # Filter and return the DataFrame for these eligible sites
    return df[df['SITE_ID'].isin(eligible_sites)]


def _get_mda8o3_daily_data(region, month, criteria=90):

    df = _filter_sites_by_observation_availability(region, month, criteria)

    # Convert OZONE concentrations to numeric
    df['OZONE'] = pd.to_numeric(df['OZONE'], errors='coerce')
    
    # Calculate 8-hour rolling mean and shift the result to center the window
    df['O3_mda8'] = df['OZONE'].rolling(window=8, min_periods=2, center=True).mean()

    # Calculate the daily max of the 8-hour average ozone for each site and date
    O3_MDA8_daily_max = df.groupby(['SITE_ID', 'DATE'])['O3_mda8'].max().reset_index()

    return O3_MDA8_daily_max


def _get_complete_valid_site_info(region, month, criteria=90):

    df_valid = _filter_sites_by_observation_availability(region, month, criteria)
    site_locations_df = _get_all_site_locations(region)
    
    # Merge updated site locations with df_dec_selected
    merged_df = pd.merge(df_valid[['SITE_ID']].drop_duplicates(), site_locations_df, on='SITE_ID', how='left')

    if region=="ENA" and month==7:
        # locate the row to update
        row_index = merged_df.loc[merged_df['SITE_ID'] == 'HOW191'].index[0]

        # update the row with new values
        merged_df.loc[row_index, 'Latitude'] = 45.203963
        merged_df.loc[row_index, 'Longitude'] = -68.740041

    if region=="EAS":
        # Drop sites with nan Longitude or Latitude (cannot find)
        merged_df = merged_df.dropna(subset=['Longitude', 'Latitude'])

    # Define functions for region and grid index calculations, assuming required libraries and objects are defined
    def calculate_region(lat, lon):
        """Given lat/lon coordinates, return the IPCC AR6 region those coordinates lie in"""
        lat_array = np.atleast_1d(lat)
        lon_array = np.atleast_1d(lon)
        return regionmask.defined_regions.ar6.land.mask(lon_array, lat_array).values.item()

    def get_c48_index(lat, lon, index_type, grid=dst_grid):
        """Given lat/lon coordinates, return c48 coordinates of a given type ('nf', 'Ydim', or 'Xdim')"""
        [nf, Ydim, Xdim] = gcgridobj.cstools.find_index(lat, lon, grid, jitter_size=0.0)
        if index_type == "nf":
            return nf.item()
        elif index_type == "Ydim":
            return Ydim.item()
        elif index_type == "Xdim":
            return Xdim.item()

    # Apply calculations for region and grid indices
    for index_type in ["nf", "Ydim", "Xdim"]:
        merged_df[f'{index_type}_idx'] = merged_df.apply(lambda x: get_c48_index(x['Latitude'], x['Longitude'], index_type=index_type), axis=1)

    # Assuming regionmask and gcgridobj are properly defined and imported elsewhere
    merged_df['Region'] = merged_df.apply(lambda x: calculate_region(x['Latitude'], x['Longitude']), axis=1)

    # Create final DataFrame with all site information
    site_info_data = {
        'SITE_ID': merged_df['SITE_ID'].tolist(),
        'Latitude': merged_df['Latitude'].tolist(),
        'Longitude': merged_df['Longitude'].tolist(),
        'IPCC AR6 region': merged_df['Region'].tolist(),
        'nf': merged_df['nf_idx'].tolist(),
        'Ydim': merged_df['Ydim_idx'].tolist(),
        'Xdim': merged_df['Xdim_idx'].tolist()
    }
    site_info = pd.DataFrame(site_info_data)

    return site_info

def get_observed_daily_mda8o3_ar6(region, month, criteria=90):

    ar6_region = {
        "ENA": 5,
        "EAS": 35,
        "WCE": 17,
    }

    O3_mda8_data = _get_mda8o3_daily_data(region, month, criteria)
    site_info = _get_complete_valid_site_info(region, month, criteria)

    mda8o3_ar6 = pd.merge(O3_mda8_data, site_info,
                    how='left', left_on=['SITE_ID'], right_on=['SITE_ID'])
                    
    mda8o3_ar6 = mda8o3_ar6[mda8o3_ar6['IPCC AR6 region']==ar6_region[region]]

    return mda8o3_ar6


def get_observation_mask(sitemean_mda8o3_ar6):
    """From a list of observations with c48 coordinates, create an xarray 
    mask where these observations exist.

    Args:
        sitemean_mda8o3_ar6 (pd.DataFrame): pandas dataframe with site information (incl. cube-sphere coordinates)

    Returns:
        observation_mask (xr.Dataset): mask of observations

    """

    cs48_example_clean = cs48_example['SpeciesConc_O3'].isel(lev=0,time=0)

    # Create an empty xarray dataset
    nf = np.asarray(cs48_example_clean.nf)
    Ydim = np.asarray(cs48_example_clean.Ydim)
    Xdim = np.asarray(cs48_example_clean.Xdim)

    observation_mask = xr.Dataset(
        coords={"nf": nf, "Ydim": Ydim, "Xdim": Xdim}
    )

    # Initialize variables for each type of data
    observation_mask["O3_mda8"] = (("nf", "Ydim", "Xdim"), np.full((len(nf), len(Ydim), len(Xdim)), np.nan))

    # Iterate over each row in the sitemean dataframe
    for index, row in sitemean_mda8o3_ar6.iterrows():
        observation_mask["O3_mda8"][int(row['nf']), int(row['Ydim']), int(row['Xdim'])] = row["O3_mda8"]

    return observation_mask


def get_masked_model_mda8o3(ds_ref, observation_mask, month):
    """Select GCHP model values where there are observations        
    """
    da_ref = ds_ref['SpeciesConc_O3'].sel(time=ds_ref.time.dt.month.isin(month))
    da_ref = (da_ref.isel(lev=0, drop=True)).where(observation_mask['O3_mda8']>0) 
    return da_ref


def crop_regionmask_ar6_c48(c48_da, region_num):
    mask_c48 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.ar6.all.mask_3D.c48.nc")

    c48_da_masked = c48_da.where(mask_c48.sel(region=region_num))

    return c48_da_masked


def mask_ocean_c48(c48_da):
    mask_c48 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask_3D.c48.nc")
    c48_da_masked = c48_da.where((mask_c48))
    return c48_da_masked