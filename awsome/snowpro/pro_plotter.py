################################################################################
# Copyright 2022 Avalanche Warning Service Tyrol                               #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray
import time

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.dates import DateFormatter
from snowpacktools.snowpro import snowpro, pro_helper, instability_rfm_mayer


def plot_snp_evo(config, DATETIME_STR=None):
    """Plots snowpack evolution (PRO-file). Different variables or grain type can be visualized and overlayed.
    
    Arguments:
        path_to_pro (str):    Path to PRO file to be plotted
        output_dir (str):     
        DATETIME_STR (str):
        var: optional   Can be used for visualizing other variables than grain type
        res (str):
        second_var (str):
        COLOR_SCHEME (str):
        DATE_RANGE (list):
    Returns:
        Figure
    """

    var          = config.get('SNOWPRO-EVO', 'VAR')
    second_var   = config.get('SNOWPRO-EVO','SECOND_VAR')
    COLOR_SCHEME = config.get('SNOWPRO','COLOR_SCHEME')
    DATE_RANGE   = [config.get('SNOWPRO-EVO', 'START_DATE'), config.get('SNOWPRO-EVO', 'END_DATE')]
    try:
        output_name = config.get('SNOWPRO', 'OUTPUT_NAME')
    except:
        output_name = 'NONE'

    """Read .pro file"""
    start_time= time.time()
    profs, meta_dict = snowpro.read_pro(config.get('SNOWPRO','PRO_FILE_PATH'), res=config.get('SNOWPRO-EVO', 'RESOLUTION'), keep_soil=False, consider_surface_hoar=True)

    """Filter for certain resolution and time frame"""
    w, __ = pro_helper.set_resolution(config.get('SNOWPRO-EVO', 'RESOLUTION'))
    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, _, _, _ = pro_helper.get_grain_type_colors(COLOR_SCHEME)
    RANGE_DICT = pro_helper.get_range_dict()

    """Color map and preprocessing"""
    rta = 0.8
    if var=='grain_type':
        col_dict_labels     = dict(zip(LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE))
        hatches_dict_labels = dict(zip(LABELS_GRAIN_TYPE, HATCHES_GRAIN_TYPE))
    else:
        if var == 'Punstable':
            profs     = instability_rfm_mayer.calc_punstable(profs)
            cmap_var  = pro_helper.get_Punstable_cmap()
            var_ticks = np.arange(0,1.1,0.1)
        else:
            if var == 'Sk38' or var == 'Sn38':
                cmap_var  = pro_helper.get_sk38_cmap()
                var_ticks = np.arange(0,1.6,0.1)
            else:
                cmap_var = plt.get_cmap('plasma_r')
        clev_var = np.linspace(RANGE_DICT[var][0],RANGE_DICT[var][1],100) # 11
        # cnorm_var = BoundaryNorm(boundaries=clev_var, ncolors=cmap_var.N, clip=True)

    plot_second_var  = False
    var_alpha        = 1
    second_vars      = ['Sk38','Sn38','Punstable']
    hatch_second_var = ''
    if second_var in second_vars:
        plot_second_var = True
        var_alpha=0.33
        if second_var == 'Sk38' or second_var == 'Sn38':
            cmap_var2        = pro_helper.get_sk38_cmap()
            second_var_ticks = np.arange(0,1.6,0.1)
        elif second_var == 'Punstable':
            profs            = instability_rfm_mayer.calc_punstable(profs)
            cmap_var2        = pro_helper.get_Punstable_cmap()
            second_var_ticks = np.arange(0,1.1,0.1)
        else:
            #cmap_var2  = plt.get_cmap('gist_gray')
            cmap_var2 = pro_helper.get_whiteout_cmap(reverse=True)
            
        clev_var2 = np.linspace(RANGE_DICT[second_var][0],RANGE_DICT[second_var][1],100) # 11 discrete colorbar
        # cnorm_var2 = BoundaryNorm(boundaries=clev_var2, ncolors=cmap_var2.N, clip=False)

    """Visualization"""
    if DATETIME_STR==None:
        # fig, (ax_cbar,ax) = plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios': [0.5,11]})
        fig, ax = plt.subplots(1,1,figsize=(9.5,5))
    else:
        fig, (ax,ax_prof) = plt.subplots(1,2,figsize=(11,5),sharey=True,gridspec_kw={'width_ratios': [11,4]})
    # ax_cbar.axis('off')

    h_max = []
    dates = []
    for i, ts in enumerate(profs):
        prof = profs[ts]
        dates.append(ts)
        if len(prof['height']>0):
            h_max.append(prof['height'][-1])
                
            if var=='grain_type':
                cols   =[]
                hatches=[]
                for row in range(0,len(prof['graintype'])):
                    cols.append(col_dict_labels[prof['graintype'][row][0]])
                    hatches.append(hatches_dict_labels[prof['graintype'][row][0]])
                ax.bar(ts, prof['thickness'], width=w, bottom=prof['bottom'], align='edge', color=cols, hatch=hatches, alpha=var_alpha) # label=labels[i])
            else:
                thickness = np.where(prof['RTA']>=rta, prof['thickness'], np.nan)
                bottom    = np.where(prof['RTA']>=rta, prof['bottom'],    np.nan)

                var_data = (prof[var]-RANGE_DICT[var][0])/(RANGE_DICT[var][1]-RANGE_DICT[var][0])
                cols     = cmap_var(var_data)
                ax.bar(ts, prof['height'][-1], width=w, bottom=0, align='edge', color='lightgrey', alpha=1)
                ax.bar(ts, thickness, width=w, bottom=bottom, align='edge', color=cols, alpha=1)
            
            if plot_second_var:
                """Filter data with RTA"""
                thickness = np.where(prof['RTA']>=rta, prof['thickness'], np.nan)
                bottom    = np.where(prof['RTA']>=rta, prof['bottom'],    np.nan)
                hatches   = np.where(prof['RTA']>=rta, hatch_second_var, np.nan)

                var_data2 = np.where(prof['RTA']>=rta, prof[second_var],  np.nan)
                var_data2 = (var_data2-RANGE_DICT[second_var][0])/(RANGE_DICT[second_var][1]-RANGE_DICT[second_var][0])
                cols2 = cmap_var2(var_data2)
                ax.bar(ts, thickness, width=w, bottom=bottom, align='edge', color=cols2, hatch=hatches, alpha=1)
        else:
            h_max.append(0)

    """Line along snow surface"""
    ax.plot(dates,h_max,ds='steps-post',lw=0.8,color='black', ls='--', alpha=0.67)

    """Hardness profile to the right"""
    if DATETIME_STR!=None:
        ax_prof, DATETIME_STR = plot_single_profile(config,ax=ax_prof)
        datetime_format = '%Y-%m-%dT%Hh%M'
        time_of_profile = datetime.strptime(DATETIME_STR, datetime_format)
        #ax.axvline(x=time_of_profile,ymin=-0.1, ymax=1.1, color='black', lw=1.5, ls='--')

    """COLORBAR (Norm, bins, formatter, ticks - lots of stuff to make colorbar look nice)"""
    if second_var!='NONE':
         # - Colorbar for second layer - # 
        n_var = 9
        lulu = np.zeros((n_var,n_var))
        for nn in range(0,n_var):
            lulu[nn, :] = np.nan # nn
        contf = ax.contourf(lulu, cmap=cmap_var2, levels=clev_var2, hatches=hatch_second_var)
        # cbar2 = fig.colorbar(contf,ax=ax_cbar, location='left', ticks=second_var_ticks, pad=0, fraction=1, aspect=30, extend='both') # shrink=0.7, ax=[axes[1],axes[3], axes[5]]
        cbar2 = fig.colorbar(contf,ax=ax, location='left', ticks=second_var_ticks, pad=0.02, extend='both') # shrink=0.7
        cbar2.set_label(second_var)

        """Modify grain type legend for SARPGR"""
        LABELS_GRAIN_TYPE  = ['PP(gp), DF','SH, DH','FC(xr), RG','MF(cr), IF']
        COLORS_GRAIN_TYPE  = ['#ffde00','#95258f','#dacef4','#d5ebb5']
        HATCHES_GRAIN_TYPE = ['','','','']
        pro_helper.add_custom_legend(ax, LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, x=0.188, y=1.03, width=0.028, height=0.025, spacing=0.17, alpha=var_alpha)
    else:
        if var=='grain_type':
            pro_helper.add_custom_legend(ax, LABELS_GRAIN_TYPE[1:], COLORS_GRAIN_TYPE[1:], HATCHES_GRAIN_TYPE[1:], x=0.015, y=1.03, width=0.028, height=0.025, spacing=0.092, alpha=var_alpha)
        else:
            n_var = 9
            lulu = np.zeros((n_var,n_var))
            for nn in range(0,n_var):
                lulu[nn, :] = np.nan # nn
            contf = ax.contourf(lulu,cmap=cmap_var,levels=clev_var) #extend='max'
            # cbar = fig.colorbar(contf,ax=ax_cbar, location='left', ticks=var_ticks, fraction=1, extend='both')
            cbar = fig.colorbar(contf,ax=ax, location='left', ticks=var_ticks, pad=0.02, extend='both')
            cbar.set_label(var)
    
    """Axes and labels""" 
    if DATE_RANGE[0] == 'NONE':
        ax.set_xlim(dates[0],dates[-1])
    else:
        DATETIME_FORMAT  = '%Y-%m-%d' # +01:00
        d0 = datetime.strptime(DATE_RANGE[0], DATETIME_FORMAT)
        d1 = datetime.strptime(DATE_RANGE[1], DATETIME_FORMAT)
        ax.set_xlim(d0,d1)
    
    if config.get("SNOWPRO","HEIGHT_MAX") != "NONE":
        ax.set_ylim(0,float(config.get("SNOWPRO","HEIGHT_MAX")))
    else:
        ax.set_ylim(0,1.1*np.max(h_max))

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_tick_params(labelright=True)
    ax.set_ylabel("height / cm")
    
    date_fmt = DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(date_fmt)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    """Include Meta data in top left corner and save figure"""
    meta_x = 0.015
    meta_y = 0.975
    header_str   = 'Location:' + '\nElevation:' + '\nSlope Angle:' + '\nAspect:'
    header_str_2 = meta_dict['StationName'] + '\n' + meta_dict['Altitude'] + 'm\n' + str(int(float(meta_dict['SlopeAngle']))) + '°\n' + str(int(float(meta_dict['SlopeAzi'])))  + '°'
    #ax.text(meta_x,        meta_y, header_str,   horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=10) # ma='left'
    #ax.text(meta_x + 0.12, meta_y, header_str_2, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=10) # ma='left'
    
    """Save figure"""
    if output_name == 'NONE':
        filename = config.get('SNOWPRO','PRO_FILE_PATH').split("/")[-1].split(".")[0]
        if DATETIME_STR==None:
            fig_title = filename + '_snp_evo.png'
        else:
            fig_title = filename + '_snp_evo_and_profile.png'
    else:
        fig_title = output_name
    fig.tight_layout()
    print(f'[i]  Saving figure "{fig_title}" to "{config.get("SNOWPRO","OUTPUT_DIR")}".')
    fig.savefig(os.path.join(config.get("SNOWPRO","OUTPUT_DIR"),fig_title), facecolor='w', edgecolor='w',
                format='png', dpi=150, bbox_inches='tight')

    print('[i]  Visualization of snowpack evolution completed in {}s'.format(time.time()-start_time))


def plot_single_profile(config, ax=None):
    """Plots snow profile of PRO file @DATETIME (shows hardness profile + grain type).
    
    Arguments:
        path_to_pro: path to PRO file to be plotted
    Returns:
        Figure
    """
    DATETIME_STR = config.get('SNOWPRO-PROF', 'DATETIME')
    COLOR_SCHEME=config.get('SNOWPRO','COLOR_SCHEME')

    profs, meta_dict = snowpro.read_pro(config.get('SNOWPRO','PRO_FILE_PATH'))

    LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE, HATCHES_GRAIN_TYPE, LABELS_GRAIN_TYPE_BAR, COLORS_GRAIN_TYPE_BAR, HATCHES_GRAIN_TYPE_BAR = pro_helper.get_grain_type_colors(COLOR_SCHEME)

    """Get closest profile of PRO file to DATETIME"""
    date_found = 0
    datetime_format  = '%Y-%m-%dT%Hh%M'
    
    ts = datetime.strptime(DATETIME_STR, datetime_format)
    if ts in profs.keys():
        print('[i]  Timestamp {} found. Hardness profile will be plotted.'.format(DATETIME_STR))
    else:
        ts = list(profs)[-1]
        print('[i]  Timestamp {} not found. Hardness profile will be plotted for last timestamp.'.format(DATETIME_STR))
        DATETIME_STR = datetime.strftime(ts, datetime_format)

    """Use current ts from now on"""
    prof = profs[ts]

    hand_hardness_param_needed = True
    if np.max(abs(prof['hand hardness'])) > 10:
        hand_hardness_param_needed = False
    hand_hardness_dict, tickz_hh, tick_labels_hh = pro_helper.get_hand_hardness_N_dict()
    prof['hand_hardness_N'] = prof['hand hardness'] # Just initialisation
    ZERO_HH_VAL = 50

    if ax==None:
        fig, ax = plt.subplots(1,1,figsize=(4.5,5))
    else:
        fig=None

    col_dict_labels     = dict(zip(LABELS_GRAIN_TYPE, COLORS_GRAIN_TYPE))
    hatches_dict_labels = dict(zip(LABELS_GRAIN_TYPE, HATCHES_GRAIN_TYPE))

    n_bar = len(LABELS_GRAIN_TYPE_BAR)
    col_nums = np.arange(0,n_bar)
    col_dict = dict(zip(col_nums, COLORS_GRAIN_TYPE_BAR[::-1]))
    cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

    cols=[]
    hatches=[]
    for ilayer in range(0,len(prof['graintype'])):
        cols.append(col_dict_labels[prof['graintype'][ilayer][0]])
        hatches.append(hatches_dict_labels[prof['graintype'][ilayer][0]])
        if hand_hardness_param_needed:
            prof['hand_hardness_N'][ilayer] = hand_hardness_dict[prof['hand hardness'][ilayer]]
    bar_plot   = ax.barh(prof['bottom'], prof['hand_hardness_N'], height=prof['thickness'], align='edge', color=cols, hatch=hatches) # label=labels[i])
    bar_plot_p = ax.barh(prof['bottom'], ZERO_HH_VAL, height=prof['thickness'], align='edge', color=cols, hatch=hatches)
        

    # COLORBAR (Norm, bins, formatter, ticks - lots of stuff to make colorbar look nice)
    if fig!=None:
        lulu = np.zeros((n_bar,n_bar))
        for nn,k in enumerate(col_dict.keys()):
            lulu[nn, :] = np.nan
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

        norm = BoundaryNorm(norm_bins, n_bar, clip=True)
        fmt = FuncFormatter(lambda x, pos: LABELS_GRAIN_TYPE_BAR[::-1][norm(x)])
        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2

        # contf = ax.contourf(lulu,cmap=cmap,norm=norm,levels=norm_bins) # just for colorbar
        contf = ax.contourf(lulu,cmap=cmap,norm=norm,levels=norm_bins,hatches=HATCHES_GRAIN_TYPE_BAR[::-1]) # just for colorbar
        cbar = fig.colorbar(contf, format=fmt, ticks=tickz,location='left', pad=0.04) # shrink=0.7, ax=[axes[1],axes[3], axes[5]]
        cbar.ax.grid(visible=False)

    # Temperature axis
    ax_t = ax.twiny()
    ax_t.plot(prof['temperature'], prof['height'], color='#DC143C',lw=1)
    ax_t.grid(visible=False)
    ax_t.xaxis.tick_bottom()
    ax_t.xaxis.set_label_position("bottom")
    ax_t.set_xlabel("snow temperature / °C", color='#DC143C')
    ax_t.tick_params(axis='x', colors='#DC143C')
    ax_t.set_xlim(min(prof['temperature'])-2,0)

    # AXES
    XLIM = [-1100,50] 
    ax.set_xlim(XLIM)
    if fig!=None:
        if config.get("SNOWPRO","HEIGHT_MAX") != "NONE":
            ax.set_ylim(0,float(config.get("SNOWPRO","HEIGHT_MAX")))
        else:
            ax.set_ylim(0,1.1*prof['height'][-1])
        ax.set_ylabel("height / cm")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
        
    # fig.canvas.draw()
    # tickz = ax.get_xticks()
    # tick_labels = ax.get_xticklabels()
    tickz       = [-1000,-750,-500,-250,0]
    tick_labels = [1000,750,500,250,0]
    ax.set_xticks(tickz)
    ax.set_xticklabels(tick_labels)
    
    ax_hh = ax.twiny()
    ax_hh.set_xlim(XLIM)
    ax_hh.grid(visible=False)
    ax_hh.xaxis.tick_top()
    ax_hh.xaxis.set_label_position("top")
    # tickz       = [-1000,-500,-250,-100,-20]
    # tick_labels = ['K','P','1F','4F','F']
    ax_hh.set_xticks(tickz_hh)
    ax_hh.set_xticklabels(tick_labels_hh)
    ax_hh.tick_params(axis="x",direction="in", pad=-18)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("hand hardness / N")

    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Include Meta data in top left corner and save figure
    if fig!=None:
        meta_x = 0.04
        meta_y = 0.92
        header_str   = 'Location:' + '\nElevation:' + '\nSlope Angle:' + '\nAspect:'
        header_str_2 = meta_dict['StationName'] + '\n' + meta_dict['Altitude'] + 'm\n' + str(int(float(meta_dict['SlopeAngle']))) + '°\n' + str(int(float(meta_dict['SlopeAzi'])))  + '°'
        ax.text(meta_x,       meta_y, header_str,    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=10)
        ax.text(meta_x + 0.28, meta_y, header_str_2, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=10)

        # --- Save figure --- #
        filename = config.get('SNOWPRO','PRO_FILE_PATH').split("/")[-1].split(".")[0]
        fig_title = filename + '_snow_profile.png'
        fig.tight_layout()
        print(f'[i]  Saving figure "{fig_title}" to "{config.get("SNOWPRO","OUTPUT_DIR")}".')
        fig.savefig(os.path.join(config.get("SNOWPRO","OUTPUT_DIR"),fig_title), facecolor='w', edgecolor='w',
                    format='png', dpi=150)
    else:
        ax.text(0.04,0.93,DATETIME_STR,horizontalalignment='left',
                verticalalignment='top', fontsize=10, transform=ax.transAxes) # ma='left'
        return ax, DATETIME_STR