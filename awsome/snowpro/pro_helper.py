################################################################################
# Copyright 2022 Avalanche Warning Service Tyrol                               #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import datetime
import numpy as np
import pandas as pd
import time

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
from matplotlib.patches import Rectangle

def get_pro_code_dict():
    """NUmber in PRO file and corresponding variable."""

    pro_code_dict ={"0500": "date",
                    "0501": "height",                           # height [> 0: top, < 0: bottom of elem.] (cm)
                    "0502": "density",                          # element density (kg m-3)
                    "0503": "temperature",                      # element temperature (degC)
                    "0504": "element ID (1)",
                    "0506": "lwc",                              # liquid water content by volume (%)
                    "0508": "dendricity (1)",
                    "0509": "sphericity (1)",
                    "0510": "coordination number (1)",
                    "0511": "bond size (mm)",
                    "0512": "grain size (mm)",
                    "0513": "grain type (Swiss Code F1F2F3)",
                    "0514": "grain type, grain size (mm), and density (kg m-3) of SH at surface", # 0514,3,660,0.9,100 vs. 0514,3,-999,-999.0,-999.0
                    "0515": "ice volume fraction (%)",
                    "0516": "air volume fraction (%)",
                    "0517": "stress in (kPa)",
                    "0518": "viscosity (GPa s)",
                    "0519": "soil volume fraction (%)",
                    "0520": "temperature gradient (K m-1)",
                    "0521": "thermal conductivity (W K-1 m-1)",
                    "0522": "absorbed shortwave radiation (W m-2)",
                    "0523": "viscous deformation rate",         # (1.e-6 s-1)
                    "0531": "deformation rate stability index Sdef",
                    "0532": "Sn38",                             # natural stability index Sn38
                    "0533": "Sk38",                             # stability index Sk38
                    "0534": "hand hardness",                    # either (N) or index steps (1)",
                    "0535": "optical equivalent grain size (mm)",
                    "0540": "bulk salinity (g/kg)",
                    "0541": "brine salinity (g/kg)",
                    "0601": "snow shear strength (kPa)",
                    "0602": "grain size difference (mm)",
                    "0603": "hardness difference (1)",
                    "0604": "RTA",                              # "Structural stability" index or with SNP-HACKING "Relative threshold sum approach (RTA)"
                    "0605": "inverse texture index ITI (Mg m-4)",
                    "0606": "critical cut length (m)"}


    """Set up list of variabels of interest"""
    var_codes =['0501','0502','0503','0506','0508','0523',
                '0509','0511','0512','0513','0515','0516',
                '0521','0535','0517','0532','0533','0534',
                '0601','0514','0604','0606']
    return pro_code_dict, var_codes


def slf_graintypes_to_ICSSG(graintypes):
    """Transform SLF code to ICSSG abbrevations.
    Arguments:
        graintype (int): Three digit code (eg. 330)
    
    Output: 
        graintype_ICSSG (list): List with three entries ['primary gt', 'secondary gt', 'cycle']

    ---- IACS-CODE ----
    'PPgp': 0  Graupel
    'PP':   1  Precipitation particles
    'DF':'  2  Decomposing fragmented
    'RG':   3  Rounded Grains
    'FC':   4  Faceted Crystals
    'DH':   5  Depth Hoar
    'SH':   6  Surface Hoar
    'MF':   7  Melt Forms
    'MFcr'  7b Melt Freeze Crust
    'IF':   8  Ice formations
    'FCxr'  9  Faceted, rounded Crystals
    """

    grain_dic = { 0:'PPgp', 1: 'PP', 2 :'DF', 3 :'RG', 4: 'FC', 5:'DH',6:'SH',7:'MF',8:'IF',9:'FCxr'}
    cycle_dic = {0:'dry', 1: 'mel', 2:'rfr'} # dry snow, first time melting snow or already refrozen snow

    graintypes_ICSSG = []
    for graintype in graintypes:
        if np.isnan(graintype):
            return ['-999','-999','-999']
        graintype = str(int(graintype))

        digits = [int(x) for x in graintype]
        graintype_ICSSG = []

        """Convert graintypes"""
        for element in digits[:2]:
            graintype_ICSSG.append(grain_dic[element])

        """Convert cycle"""
        for element in digits[2:]:
            graintype_ICSSG.append(cycle_dic[element])
        
        if len(graintype_ICSSG)==3: 
            if graintype_ICSSG[2] == 'rfr':
                graintype_ICSSG[0] = 'MFcr'
        elif len(graintype_ICSSG)==1:
                graintype_ICSSG.append(graintype_ICSSG[0])
                graintype_ICSSG.append('dry')
        elif len(graintype_ICSSG)==2:
                graintype_ICSSG.append('dry')
        else:
            raise ValueError('Lenght of graintype list is invalid!')

        graintypes_ICSSG.append(graintype_ICSSG)
    return graintypes_ICSSG


def slf_graintype_to_ICSSG(graintype):
    """Transform SLF code to ICSSG abbrevations.
    Arguments:
        graintype (int): Three digit code (eg. 330)
    
    Output: 
        graintype_ICSSG (list): List with three entries ['primary gt', 'secondary gt', 'cycle']

    ---- IACS-CODE ----
    'PPgp': 0  Graupel
    'PP':   1  Precipitation particles
    'DF':'  2  Decomposing fragmented
    'RG':   3  Rounded Grains
    'FC':   4  Faceted Crystals
    'DH':   5  Depth Hoar
    'SH':   6  Surface Hoar
    'MF':   7  Melt Forms
    'MFcr'  7b Melt Freeze Crust
    'IF':   8  Ice formations
    'FCxr'  9  Faceted, rounded Crystals
    """

    if np.isnan(graintype):
        return ['-999','-999','-999']
    graintype = str(int(graintype))

    grain_dic = { 0:'PPgp', 1: 'PP', 2 :'DF', 3 :'RG', 4: 'FC', 5:'DH',6:'SH',7:'MF',8:'IF',9:'FCxr'}
    cycle_dic = {0:'dry', 1: 'mel', 2:'rfr'} # dry snow, first time melting snow or already refrozen snow

    digits = [int(x) for x in graintype]
    graintype_ICSSG = []

    """Convert graintypes"""
    for element in digits[:2]:
        graintype_ICSSG.append(grain_dic[element])

    """Convert cycle"""
    for element in digits[2:]:
        graintype_ICSSG.append(cycle_dic[element])
    
    if len(graintype_ICSSG)==3: 
        if graintype_ICSSG[2] == 'rfr':
            graintype_ICSSG[0] = 'MFcr'
    elif len(graintype_ICSSG)==1:
            graintype_ICSSG.append(graintype_ICSSG[0])
            graintype_ICSSG.append('dry')
    elif len(graintype_ICSSG)==2:
            graintype_ICSSG.append('dry')
    else:
        raise ValueError('Lenght of graintype list is invalid!')
    
    return graintype_ICSSG


def get_grain_type_colors(COLOR_SCHEME):
    """Available color schemes are European (IACS2) and two problem oriented schemes from the Canadian avalanche community (SARP)."""

    LABELS_GRAIN_TYPE        = ['-999','PP','DF','PPgp','SH','DH','FC','FCxr','RG','MF','MFcr','IF']
    COLORS_GRAIN_TYPE_IACS2  = ['white','#00FF00','#228B22','#696969','#FF00FF','#0000FF','#ADD8E6','#ADD8E6','#FFB6C1','#FF0000','#FF0000','#00FFFF']
    COLORS_GRAIN_TYPE_SARP   = ['white','#ffde00','#f1f501','#ffff33','#ff0000','#0078ff','#b2edff','#dacef4','#ffccd9','#d5ebb5','#addd8e','#a3ddbb']
    COLORS_GRAIN_TYPE_SARPGR = ['white','#ffde00','#ffde00','#ffde00','#95258f','#95258f','#dacef4','#dacef4','#dacef4','#d5ebb5','#d5ebb5','#d5ebb5']
    COLORS_GRAIN_TYPE_GREY   = ['white','#dcdcdc','#dcdcdc','#dcdcdc','#808080','#808080','#d3d3d3','#d3d3d3','#d3d3d3','#c0c0c0','#c0c0c0','#c0c0c0']
    HATCHES_GRAIN_TYPE_IACS2 = ['','','','','','','','','','','|||','']
    HATCHES_GRAIN_TYPE_SARP  = ['','','','','','','','','','','','']

    """INFO: Difference of lists above and _BAR below is FCxr and FC combined and no addtional spot for '-999'"""
    LABELS_GRAIN_TYPE_BAR        = ['PP','DF','PPgp','SH','DH','FC(xr)','RG','MF','MFcr','IF']
    COLORS_GRAIN_TYPE_BAR_IACS2  = ['#00FF00','#228B22','#696969','#FF00FF','#0000FF','#ADD8E6','#FFB6C1','#FF0000','#FF0000','#00FFFF']
    COLORS_GRAIN_TYPE_BAR_SARP   = ['#ffde00','#f1f501','#ffff33','#ff0000','#0078ff','#b2edff','#ffccd9','#d5ebb5','#addd8e','#a3ddbb']
    COLORS_GRAIN_TYPE_BAR_SARPGR = ['#ffde00','#ffde00','#ffde00','#95258f','#95258f','#dacef4','#dacef4','#d5ebb5','#d5ebb5','#d5ebb5']
    COLORS_GRAIN_TYPE_BAR_GREY   = ['#dcdcdc','#dcdcdc','#dcdcdc','#808080','#808080','#d3d3d3','#d3d3d3','#c0c0c0','#c0c0c0','#c0c0c0']
    HATCHES_GRAIN_TYPE_BAR_IACS2 = ['','','','','','','','','|||','']
    HATCHES_GRAIN_TYPE_BAR_SARP  = ['','','','','','','','','','']

    if COLOR_SCHEME == 'IACS2':
        COLORS_GRAIN_TYPE        = COLORS_GRAIN_TYPE_IACS2
        HATCHES_GRAIN_TYPE       = HATCHES_GRAIN_TYPE_IACS2
        COLORS_GRAIN_TYPE_BAR    = COLORS_GRAIN_TYPE_BAR_IACS2
        HATCHES_GRAIN_TYPE_BAR   = HATCHES_GRAIN_TYPE_BAR_IACS2
    elif COLOR_SCHEME == 'SARP':
        COLORS_GRAIN_TYPE        = COLORS_GRAIN_TYPE_SARP
        HATCHES_GRAIN_TYPE       = HATCHES_GRAIN_TYPE_SARP
        COLORS_GRAIN_TYPE_BAR    = COLORS_GRAIN_TYPE_BAR_SARP
        HATCHES_GRAIN_TYPE_BAR   = HATCHES_GRAIN_TYPE_BAR_SARP
    elif COLOR_SCHEME == 'GREY':
        COLORS_GRAIN_TYPE        = COLORS_GRAIN_TYPE_GREY
        HATCHES_GRAIN_TYPE       = HATCHES_GRAIN_TYPE_SARP
        COLORS_GRAIN_TYPE_BAR    = COLORS_GRAIN_TYPE_BAR_GREY
        HATCHES_GRAIN_TYPE_BAR   = HATCHES_GRAIN_TYPE_BAR_SARP
    else:
        COLORS_GRAIN_TYPE        = COLORS_GRAIN_TYPE_SARPGR
        HATCHES_GRAIN_TYPE       = HATCHES_GRAIN_TYPE_SARP
        COLORS_GRAIN_TYPE_BAR    = COLORS_GRAIN_TYPE_BAR_SARPGR
        HATCHES_GRAIN_TYPE_BAR   = HATCHES_GRAIN_TYPE_BAR_SARP

    return LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR, HATCHES_GRAIN_TYPE_BAR
    

def add_custom_legend(ax, labels, colors, hatches, x, y, width, height, spacing, alpha):
    for i, (color, label) in enumerate(zip(colors, labels)):
        rect = Rectangle((x + i * spacing, y), width, height, transform=ax.transAxes, clip_on=False, facecolor=color, hatch=hatches[i], alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + 1.2*width + i * spacing, y, label, transform=ax.transAxes)


def get_whiteout_cmap(reverse=False):
    """Helpful colormap for indices like SK38 or RTA."""
    # cmap_var2  = plt.get_cmap('gist_gray_r')
    # cmap_var2  = plt.get_cmap('plasma_r') # plasma_r, YlOrRd
    # upper = cm.plasma_r(np.arange(240))

    # cmap_gist_rainbow =  cm.get_cmap('gist_rainbow', 256)
    # cmap = cmap_gist_rainbow(np.arange(150))
    # cmap = cmap[0:85]
    # cmap = ListedColormap(cmap, name='sk38', N=cmap.shape[0])

    if reverse:
        lower = cm.gist_heat(np.arange(240))
        upper = np.ones((16,4))
    else:
        upper = cm.gist_heat_r(np.arange(240))
        lower = np.ones((16,4))
    cmap  = np.vstack(( lower, upper ))
    cmap  = ListedColormap(cmap, name='whiteout_cmap', N=cmap.shape[0])
    return cmap


def get_sk38_cmap2(colors):
    """Create a custom colormap with smooth transitions between the given colors."""

    colors = [(0.0, (1, 0, 0)), (0.3, (1, 1, 0)), (0.7, (0, 1, 0)), (1.0, (0, 0, 1))]
    
    cmap_dict = {'red': [], 'green': [], 'blue': []}
    for i in range(len(colors) - 1):
        pos_start, color_start = colors[i]
        pos_end, color_end = colors[i + 1]
        cmap_dict['red'].append((pos_start, color_start[0], color_end[0]))
        cmap_dict['green'].append((pos_start, color_start[1], color_end[1]))
        cmap_dict['blue'].append((pos_start, color_start[2], color_end[2]))

    cmap = LinearSegmentedColormap('custom_cmap', cmap_dict)
    return cmap


def get_sk38_cmap():
    """Create a custom colormap with smooth transitions between the given colors."""
    c0 = 'red'
    c1 = 'darkorange'
    c2 = 'yellow'
    c3 = 'limegreen'
    colors = [(0.0, c0), (0.32, c0), (0.4, c1), (0.6, c1), (0.68, c2), (0.73, c2), (0.78, c3), (1.0, c3)]
    
    cmap = LinearSegmentedColormap.from_list('Sk38', colors=colors, N=256)
    return cmap


def get_Punstable_cmap():
    """Create a custom colormap with smooth transitions between the given colors."""
    c4 =  '#d1001f' # '#C21807'
    #c3 = '#ED2939'
    c3 = '#C21807'
    c2 = '#FF9F00'  # 'darkorange' '#FF9F00' #
    c1 = '#85CC6F' # '#A0E989'
    c0 = '#136207'
    colors = [(0.0, c0), (0.03, c0), (0.05, c1), (0.4, c1), (0.45, c2), (0.65, c2), (0.7, c3), (0.8, c3), (0.85, c4), (1.0, c4)]
    
    cmap = LinearSegmentedColormap.from_list('Sk38', colors=colors, N=256)
    return cmap


def get_hand_hardness_N_dict():
    """Dictionary linking hand hardness steps to force in N."""

    """
    Transfomration within SNOWPACK:
        -1.*(19.3*pow(EMS[e].hard, 2.4)) -> -19.3*hand_hardness^2.4
    """
    hand_hardness_dict =    {0  :0,
                            -1  :-20,
                            -1.5:-51,
                            -1.6:-51, ## BUG OF SNP?
                            -2  :-102,
                            -2.5:-174,
                            -3  :-269,
                            -3.5:-390,
                            -4  :-538,
                            -4.5:-713,
                            -5  :-918,
                            -6  :-1422}
    
    tickz_hh       = [-918,-538,-269,-102,-20]
    tick_labels_hh = ['K','P','1F','4F','F']

    return hand_hardness_dict, tickz_hh, tick_labels_hh


def get_range_dict():
    """Linking range for colormap to relevant variables."""

    RANGE_DICT =   {'Sk38':                         [0,1.5],
                    'Sn38':                         [0,1.5],
                    'RTA':                          [0.7,1],
                    'Punstable':                    [0,1],
                    'critical cut length (m)':      [0,2],
                    'density':                      [0,550],
                    'temperature':                  [-12,0],
                    'lwc':                          [0,100],
                    'temperature gradient (K m-1)': [-20,20]}
    return RANGE_DICT


def set_resolution(res):
    """Set the resolution of the time evolution plot based on input."""

    if res == '1d':
        w = 1
        hours = [6]
    elif res == '3h':
        w = 1/8
        hours = np.arange(0,24,3) # [0,3,6,9,12,15,18,21]
    elif res == '2h':
        w = 1/12
        hours = np.arange(0,24,2)
    elif res == '1h':
        w = 1/24
        hours = np.arange(0,24)
    elif res == 'avaProb':
        w = 2/3 # not perfect
        hours = [7,15]

    return w, hours


def snowpro_from_snapshot(index, variables,i_ground_surf,soil_vars):
    """ Takes a dictionary of variables (a processed .Pro file) and returns a list with of dictionaries (one dict per profile/timestamp)

    Arguments:
        index (int):        Representing the index (day) that the snowpro object should be generated for
        variables (str):    Variables to be used
        i_ground_surf:      Index of ground surface
        soil_vars (list):   List of variables with values for soil layers
    Returns:
        df (pd df):         Dataframe of snowpack at single timestamp
    """

    """Get date of current timestamp"""
    line_series = variables['date'][index].split(",")
    date_format = "%d.%m.%Y %H:%M:%S"  # default
    #date_format = "%Y-%m-%d %H:%M:%S" # use for old Fabiano simulations
    current_date = datetime.datetime.strptime(line_series[1][:-1], date_format)

    """Generate a dictionary with available data"""
    dataframe_dict = {}
    for varname in variables.keys():
        if varname == 'date' or varname == 'grain type, grain size (mm), and density (kg m-3) of SH at surface':
            continue

        line_series = variables[varname][index].split(",")
        if line_series[0] == '-999':
            """No data available for this timestamp (usually height = 0.0 and other variables do not exist)"""
            dataframe_dict[varname] = [np.nan]
        else:
            """Get nb of datapoints (the second entry in the list)"""
            nvars   = int(line_series[1])

            """Isolate the actual datapoints from the metadata (catch soil layers)"""
            if i_ground_surf>0:
                if varname == 'height_m': # height_m needs to be handled seperately for soil layers
                    if nvars==i_ground_surf+1:
                        dataframe_dict[varname] = [np.nan]
                    else:
                        datapoints = line_series[-nvars+i_ground_surf+1:]
                        dataframe_dict[varname] = list(map(float, datapoints))
                elif varname in soil_vars:
                    if nvars==i_ground_surf:
                        dataframe_dict[varname] = [np.nan]
                    else:
                        datapoints = line_series[-nvars+i_ground_surf:]
                        dataframe_dict[varname] = list(map(float, datapoints)) 
                elif varname == 'grain type (Swiss Code F1F2F3)': 
                    """!!!PRO-FILE ERROR!!! grain type (Swiss code)"""
                    if nvars==1:
                        dataframe_dict[varname] = [np.nan]
                    else:
                        datapoints = line_series[-nvars:-1]
                        dataframe_dict[varname] = list(map(float, datapoints))
                else:
                    if nvars==1:
                        dataframe_dict[varname] = [np.nan]
                    else:   
                        datapoints = line_series[-nvars:]
                        dataframe_dict[varname] = list(map(float, datapoints))

            else:
                if varname == 'grain type (Swiss Code F1F2F3)':
                    """!!!PRO ERROR!!! grain type (Swiss code) description always has additional layer at the end with SLF CODE = 0"""
                    datapoints = line_series[-nvars:-1]
                else:
                    datapoints = line_series[-nvars:]
                ##### other processing here could increase speed??? #####
                dataframe_dict[varname] = list(map(float, datapoints))

    """Generate Pandas dataframe from dictionary"""
    df = pd.DataFrame.from_dict(dataframe_dict)

    """Consider surface hoar at surface"""
    line_series = variables['grain type, grain size (mm), and density (kg m-3) of SH at surface'][index].split(",")
    if line_series[0]=='-999':
        pass
    elif line_series[2]!='-999':
        surf_hoar =  list(map(float, line_series[2:5])) 
        df_top_layer = df[-1:].copy()
        df_top_layer.reset_index(drop=True, inplace=True)

        """convert to cm! and use 2*grain_size for visualization"""
        df_top_layer.loc[0,'height_m'] = df_top_layer.loc[0,'height_m']+ 2 * surf_hoar[1]/10
        df_top_layer.loc[0,'density']  = surf_hoar[2]
        df_top_layer.loc[0,'grain type (Swiss Code F1F2F3)'] = surf_hoar[0]
        
        df = pd.concat([df,df_top_layer], ignore_index = True, sort = False)

    """Calculate thickness for each layer in current snow profile"""
    df['height_m']    = df['height_m']/100
    df['thickness_m'] = df['height_m'] # Catches first layer (thickness=height)
    i = np.arange(1,len(df['height_m']))
    df.loc[i,'thickness_m'] = df.loc[i,'height_m'].values-df.loc[i-1,'height_m'].values
    ###### - Eventually include more code to cover negative height scenarios - ######

    """Add date as one column to Pandas dataframe"""
    df['date'] = current_date

    """Turn order around that highest layer is the 'first'"""
    df = df[::-1]
    df = df.reset_index(drop=True)
    return df


def get_var_smet(path):
    """Returns variables in smet file as array of strings."""

    """Line which contains the variabels starts with the keyword 'fields'"""
    keyword = 'fields '
    file1 = open(path,'r')
    flag = 0
    index = 0

    for line in file1:
        index += 1
    
        if keyword in line:
            flag = 1
            var = line
            break
        
    if flag == 0: 
        print('keyword:',keyword , 'Not Found')
        
    """Reduce complete str-lines to single variables"""
    #var = var.split(" ")
    #var = var[12:][:-1] # Start with first element & drop las element 'n

    var = var.split("=")[1].strip()
    var = var.split(" ")
    return var