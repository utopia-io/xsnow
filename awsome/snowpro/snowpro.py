################################################################################
# Copyright 2022 Avalanche Warning Service Tyrol                               #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import sys
import time
import configparser

import numpy as np
import pandas as pd
import xarray # needed for time axis
from datetime import datetime

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# import awset
# import awsmet
from snowpacktools.snowpro import pro_helper, pro_plotter

def read_pro(path,res='1h',keep_soil=False, consider_surface_hoar=True):
    """Reads a .PRO file and returns a dictionary with timestamps as keys and values being another dictionary with
    profile parameters as keys and data as value representing the evolving state of the snowpack.
    
    Arguments:
        path (str):             String pointing to the location of the .PRO file to be read
        res (str):              temporal resolution
        keep_soil (bool):       Decide if soil layers are kept
        consider_surface_hoar (bool):   Decide if surface hoar should be added as another layer
    Returns:
        profs (dict):           Dictionary with timestamps as keys and values being another dictionary with profile parameters
        meta_dict (dict):       Dictionary with metadata of snow profile
    """

    w, hours = pro_helper.set_resolution(res)

    PRO_CODE_DICT, VAR_CODES = pro_helper.get_pro_code_dict()
    VAR_CODES_PROF = []

    """Dictionary with timestamps as keys and values being another dictionary with profile parameters as keys and data as value"""
    profs = {}
    meta_dict = {}
    
    """Open the PRO file and generate dict of variables with list of lines for each variable"""
    with open(path, "r") as f:
        file_content = f.readlines()
    
    section = '[STATION_PARAMETERS]'
    for line in file_content:
        line = line.rstrip('\n')

        if section=='[DATA]':
            if line[:4] == '0500':
                # """Check that all variable lists are same length"""
                # if not_first_timestamp:
                #     for varcode in VAR_CODES:
                #         if PRO_CODE_DICT[varcode] not in profs[ts].keys():
                #             # profs[ts][PRO_CODE_DICT[varcode]] = '-999'
                #             profs[ts][PRO_CODE_DICT[varcode]] = -999

                """Check if timestamp is of interest"""
                if int(line[-8:-6]) in hours:
                    timestamp_of_interest = True
                    ts = datetime.strptime( line.strip().split(',')[1], '%d.%m.%Y %H:%M:%S' )
                    profs[ts] = {}
                    not_first_timestamp = True
                else:
                    timestamp_of_interest = False

            elif line[:4] in VAR_CODES and timestamp_of_interest:
                if line[:4] == "0513": # 'grain type (Swiss Code F1F2F3)':
                    profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:-1],dtype=float)
                elif line[:4] == "0501":
                    height = np.array(line.strip().split(',')[2:],dtype=float)
                    if len(height) == 1 and height.item() == 0: profs[ts][PRO_CODE_DICT[line[:4]]] = np.array([])
                    else: profs[ts][PRO_CODE_DICT[line[:4]]] = height
                else:
                    profs[ts][PRO_CODE_DICT[line[:4]]] = np.array(line.strip().split(',')[2:],dtype=float)

        elif section=='[HEADER]':
            if line == '[DATA]':
                """Drop variables that are not present in header of .PRO file"""
                keys_to_drop = []
                for varcode in VAR_CODES:
                    if varcode not in VAR_CODES_PROF:
                        print('[i]  Variable ', varcode, ' is not found in the header of this .PRO file. It is dropped.')
                        keys_to_drop.append(varcode)
                
                for key in keys_to_drop:
                    VAR_CODES.pop(key)
                
                section = '[DATA]'
                timestamp_of_interest = False
                not_first_timestamp   = False
                continue
            else:
                VAR_CODES_PROF.append(line[:4])

        else: # - section=='[STATION_PARAMETERS]' - #
            if line == '[HEADER]':
                section = '[HEADER]'
                continue
            else:
                line_arr = line.split('= ')
                if len(line_arr) == 2:
                    meta_dict[line_arr[0]] = line_arr[1]


    """Check again that all variable lists are same length... (for no snow at end of season)"""
    # for varcode in VAR_CODES:
    #     if PRO_CODE_DICT[varcode] not in profs[ts].keys():
    #         profs[ts][PRO_CODE_DICT[varcode]] = -999

    """Check existence of soil layers (exist for negative height values)"""
    prof_dates    = list(profs)
    first_date    = prof_dates[0]
    soil_detected = False
    
    # if 'height' in profs[first_date].keys():
    nheight = len(profs[first_date]['height'])
    soil_vars     = []
    i_ground_surf = 0
    if nheight > 0:
        if profs[first_date]['height'][0] < 0:
            print('[i]  Soil layers detected')
            soil_detected = True
            i_ground_surf = list(profs[first_date]['height']).index(0.0)
            print('[i]  i_ground_surf: ', i_ground_surf)
            for varname in profs[first_date].keys():
                n = len(profs[first_date][varname])
                if n+1==nheight:
                    soil_vars.append(varname)
            print('[i]  Variables with soil layers: ', soil_vars)
    
    if keep_soil==False and soil_detected:
        for ts in profs.keys():
            profs[ts]['height'] = profs[ts]['height'][i_ground_surf+1:]
            for var in soil_vars:
                profs[ts][var] = profs[ts][var][i_ground_surf:]

    """Calculate thickness for each layer in current snow profile"""
    for ts in profs.keys():
        """Consider surface hoar at surface"""
        if consider_surface_hoar:
            if 'grain type, grain size (mm), and density (kg m-3) of SH at surface' in profs[ts].keys():
                surf_hoar = profs[ts]['grain type, grain size (mm), and density (kg m-3) of SH at surface']
                if not np.isscalar(surf_hoar):
                    if surf_hoar[0]!=-999:
                        for var in profs[ts].keys():
                            if var == 'grain type, grain size (mm), and density (kg m-3) of SH at surface':
                                continue
                            elif var == 'height':
                                profs[ts][var] = np.append(profs[ts][var], profs[ts][var][-1] + 2 * surf_hoar[1]/10) # or np.insert()
                            elif var == 'density':
                                profs[ts][var] = np.append(profs[ts][var], surf_hoar[2])
                            elif var == 'grain type (Swiss Code F1F2F3)':
                                profs[ts][var] = np.append(profs[ts][var], surf_hoar[0])
                            else:
                                profs[ts][var] = np.append(profs[ts][var], np.nan)

        """Calculate thickness and bottom"""
        # if 'height' in profs[ts].keys():
        if len(profs[ts]['height']) > 0:
            # profs[ts]['height']    = profs[ts]['height'] / 100 # transform to m (not used anymore)
            profs[ts]['thickness'] = profs[ts]['height'].copy() # Catches first layer (thickness=height)
            i = np.arange(1,len(profs[ts]['height']))
            profs[ts]['thickness'][i] = profs[ts]['height'][i] - profs[ts]['height'][i-1]
            profs[ts]['bottom'] = profs[ts]['height'].copy()
            profs[ts]['bottom'][0] = 0
            profs[ts]['bottom'][i] = profs[ts]['height'][i-1]
        
        """Transform SLF graintype code into ICSSG standard abbreviation"""
        if 'grain type (Swiss Code F1F2F3)' in profs[ts].keys():
            if not np.isscalar(profs[ts]['grain type (Swiss Code F1F2F3)']):
                profs[ts]['graintype'] = pro_helper.slf_graintypes_to_ICSSG(profs[ts]['grain type (Swiss Code F1F2F3)'])

        """NANs"""
        # for ts in prof['data'].keys():
        #     for var in prof['data'][ts].keys():
        #         data = prof['data'][ts][var]
        #         try: prof['data'][ts][var] = np.where((data==-999),np.nan,data)
        #         except: pass
        
        """Turn order around that highest layer is the 'first'"""
        # df = df[::-1]
        # df = df.reset_index(drop=True)

    return profs, meta_dict


def read_pro_pd(path,res='1h'):
    """Reads a .PRO file and returns a list of dataframes representing the evolving state of the snowpack.
    
    Arguments:
        path (str):             String pointing to the location of the .PRO file to be read
    Returns:
        snowpro_list (list):    List of dfs with each df representing one snow profile (one timestamp), column=layer, row=variables
        meta_dict:              Dictionary with metadata of snow profile
    """
    w, hours = pro_helper.set_resolution(res)

    PRO_CODE_DICT, VAR_CODES = pro_helper.get_pro_code_dict()
    variables = {}
    VAR_CODES.append('0500')
    PRO_CODE_DICT['0501'] = 'height_m'
    for var in VAR_CODES:
        variables[PRO_CODE_DICT[var]] = []
    
    meta_dict = {}
    
    # Open the PRO file and generate dict of variables with list of lines for each variable
    start_read_file = time.time()
    with open(path, "r") as f:
        file_content = f.readlines()
    
    section = '[STATION_PARAMETERS]'
    for line in file_content:
        line = line.rstrip('\n')

        if line == '[HEADER]':
            section = '[HEADER]'
            continue
        elif line == '[DATA]':
            section = '[DATA]'
            timestamp_of_interest = False

            # Drop variables that are not present in header of .PRO file
            keys_to_drop = []
            for key in variables:
                if len(variables[key]) == 0:
                    print('[i]  Variable ', key, ' is not found in the header of this .PRO file. It is dropped.')
                    keys_to_drop.append(key)
            
            for key in keys_to_drop:
                variables.pop(key)
            continue
        
        if section=='[STATION_PARAMETERS]':
            line_arr = line.split('= ')
            if len(line_arr) == 2:
                meta_dict[line_arr[0]] = line_arr[1]
        
        elif section=='[HEADER]':
            if line[:4] in VAR_CODES:
                variables[PRO_CODE_DICT[line[:4]]].append(line)

        else:
            if line[:4] == '0500':
                # Check that all variable lists are same length...
                n = len(variables['date'])
                for key in variables:
                    if (n-len(variables[key]))==1:
                        variables[key].append('-999')

                # Check if timestamp is of interest
                if int(line[-8:-6]) in hours:
                    timestamp_of_interest=True
                    # Add line to variable
                    variables['date'].append(line)
                else:
                    timestamp_of_interest=False

            elif line[:4] in VAR_CODES and timestamp_of_interest:
                variables[PRO_CODE_DICT[line[:4]]].append(line)

    # Check again that all variable lists are same length... (for no snow at end of season)
    n = len(variables['date'])
    for key in variables:
        if (n-len(variables[key]))==1:
            variables[key].append('-999')

    end_read_file = time.time()
    print('[i]  Reading of lines took: {}s'.format(int(end_read_file-start_read_file)))

    # Remove the header data (leave this, because it covers wrong user input with var_codes)
    for variable in variables.keys():
        try:
            variables[variable].pop(0)
        except:
            print('[i]  Attention: No values for', variable,'in your .pro file. Remove it out of var_code dictionary')

    # Check existence of soil layers (!exist for negative height values!)
    line_series = variables['height_m'][0].split(",")
    nvars = int(line_series[1])
    soil_vars     = []
    i_ground_surf = 0
    if nvars > 1:
        datapoints = line_series[-nvars:]
        datapoints = list(map(float, datapoints))
        if datapoints[0] < 0:
            print('[i]  Soil layers detected. The following variables contain values for soil:')
            i_ground_surf = datapoints.index(0.0)
            for varname in variables.keys():
                if varname != 'date':
                    line_series = variables[varname][0].split(",")
                    n = int(line_series[1])
                    if n+1==nvars:
                        soil_vars.append(varname)
            print(soil_vars)

    # Generate snow profile dataframe for each timestamp
    start_processing = time.time()
    snowpro_list = [pro_helper.snowpro_from_snapshot(i, variables, i_ground_surf, soil_vars) for i in range(len(variables['date']))]
    end_processing = time.time()
    print('[i]  Generation of dataframes took: {}s'.format(int(end_processing-start_processing)))

    # Transform SLF graintype code into ICSSG standard abbreviation
    for df in snowpro_list:
        df['graintype'] = df['grain type (Swiss Code F1F2F3)'].apply(pro_helper.slf_graintype_to_ICSSG)

        i = np.arange(0,len(df)-1)
        df['bottom'] = 0
        df.loc[i,'bottom'] = df.loc[i+1,'height_m'].values

    return snowpro_list, meta_dict


def get_smet_df(path):
    """Generates dataframe for further processing out of .smet file."""

    var = pro_helper.get_var_smet(path)

    df_smet = pd.read_csv(path, sep=" ", skiprows=18, skipinitialspace=True, names =var) 
    df_smet = df_smet.replace(-999.0, np.NaN)

    # Reduce dataframe to variables of interest
    variables_of_intrest = ['timestamp', 'TSS_mod', 'TSS_meas', 'T_bottom' ,'TSG','VW','DW','wind_trans24','VW_drift', 'MS_Wind', 'HS_mod', 'HS_meas',
                        'MS_Snow','hoar_size', 'HN72_24','HN24','MS_Rain','SWE','MS_Water']
    df_smet = df_smet.loc[:, df_smet.columns.intersection(variables_of_intrest)]

    df_smet['HS_mod']  = df_smet['HS_mod']/100
    df_smet['HS_meas'] = df_smet['HS_meas']/100
    df_smet['HN72_24'] = df_smet['HN72_24']/100
    df_smet['HN24']    = df_smet['HN24']/100
    return df_smet


def snowpro(config_file=None, pro_file=None, output_dir=None):
    """A SNOWPACK output (.pro file) visualization tool.
    
    Arguments:
        config_file (str):  Path to configuration (ini) file   
        pro_file (str):     Path to PRO file
    """

    config = configparser.ConfigParser()
    this_dir, this_filename = os.path.split(__file__)
    if config_file and os.path.exists(config_file):
        config.read(config_file)
    else:
        snowpro_template_ini_path = os.path.join(this_dir, "snowpro.ini")
        if os.path.exists(snowpro_template_ini_path): 
            config.read(snowpro_template_ini_path)
        else:
            sys.exit('[E] No configuration file available')
            
    latex_template_path = os.path.join(this_dir, "latex_template.mplstyle")
    if os.path.exists(latex_template_path):
        plt.style.use(latex_template_path)

    if not output_dir:
        output_dir = config.get('SNOWPRO','OUTPUT_DIR')
    os.makedirs(output_dir, exist_ok=True)
        
    if pro_file != None:
        config['SNOWPRO']['PRO_FILE_PATH'] = pro_file

    if config.get('SNOWPRO','PLOT_SNP_EVO') =='TRUE':
        pro_plotter.plot_snp_evo(config)
    
    if config.get('SNOWPRO','PLOT_PROFILE')=='TRUE':
        pro_plotter.plot_single_profile(config)

    if config.get('SNOWPRO', 'PLOT_SNP_EVO_AND_PROFILE')=='TRUE':
        pro_plotter.plot_snp_evo(config, DATETIME_STR=config.get('SNOWPRO-PROF','DATETIME'))



if __name__ == "__main__":
    """
    import debugpy
    debugpy.listen(5678)
    print('Waiting for debugger!')
    debugpy.wait_for_client()
    print('Attached!')
    """
    args = sys.argv[1:]

    this_dir, this_filename = os.path.split(__file__)
    snowpro_template_ini_path = os.path.join(this_dir, "snowpro.ini")
    if len(args) > 0: 
        if (os.path.isfile(args[0])):
            if len(args)>1: 
                snowpro(args[0],args[1])
            else:
                snowpro(args[0])
        else:
            name = str(args[0])
            print('[E]   File ({}) not found.'.format(name))
    else:
        if os.path.exists(snowpro_template_ini_path):
            snowpro(snowpro_template_ini_path)
        else:
            sys.exit('[E]   No configuration file available')
