################################################################################
# Copyright 2022 Avalanche Warning Service Tyrol                               #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

"""
@author: Stephanie Mayer, adapted by Michi Binder
"""

import os
import time
import joblib
import numpy as np


def calc_punstable(profs):
    """Calculate Mayer's Punstable and relevant variables and add it to profiles"""

    """Load Mayer's instability model (model was developed using Python 3.7.4 and scikit.learn version 0.22.1)"""
    model = joblib.load(os.path.join("/".join(__file__.split("/")[:-1]), "models", "RF_instability_model.sav"))


    """Calculate Punstable and further properties for all profiles and layers"""
    slopeangle   = 0 # only used for calculation of penetration depth to check whether hard MFcrusts exists with thickness > 3 cm perpendicular to slope.
    feature_list = ['viscdefrate','rcflat','sphericity','grainsize','penetrationdepth','slab_rhogs']
    
    """Empty dataframe to start stacking"""
    start_process = time.time()
    features = np.empty((0,len(feature_list),))
    iprofs   = []

    for ts in profs:
        prof = profs[ts]

        """Get features for RF model and stack"""
        if (len(prof['height'])==0): # No snow on the ground (height-array is empty, remove soil for these calculations)
            df_features_prof    = np.empty((1,len(feature_list),)) # # 1 row filled with np.nan
            df_features_prof[:] = np.nan
        else:
            df_features_prof = comp_features(prof, slopeangle)
        
        iprofs.append(len(df_features_prof))

        features = np.concatenate([features, df_features_prof], axis=0)

    print('[i]  Stacking features for Punstable RF-model (Mayer et al., 2022): {}s'.format(time.time()-start_process))
    # print('Number of timestamps: ', len(list(profs)))
    # print('Number of indices:    ', len(iprofs))

    """Calculate Punstable using RF model"""
    start_process = time.time()
    
    ## - Make sure features are in right order for model - ##
    ## features = features[['viscdefrate', 'rcflat', 'sphericity', 'grainsize', 'penetrationdepth','slab_rhogs']]
    Punstable = np.repeat(np.nan,len(features))     
    
    ## compute p_unstable for all rows except the ones that contain any NA values:
    feature_mask = ~np.isnan(features).any(axis = 1)
    Punstable[feature_mask] = model.predict_proba(features[feature_mask])[:,0]

    print('[i]  RF-model prediction: {}s'.format(time.time()-start_process))
    
    i0 = 0
    for i,ts in enumerate(profs):
        i1 = i0+iprofs[i]
        #profs[ts]['Punstable'] = df_features['Punstable'][i0:i1].values
        profs[ts]['Punstable'] = Punstable[i0:i1]
        i0=i1

    return profs

    
def comp_features(prof, slopeangle):
    ###########################################################################################################
    #####
    #####   Name:       comp_features
    #####
    #####   Purpose:     given SNOWPACK profile in readProfile format, calculate all necessary features for every single layer (from bottom to top of profile)
    #####               
    #####   Remarks:  don't change order of features in output (the RF model needs exactly this order)
    #####
    ###########################################################################################################

    strength_s    = prof['snow shear strength (kPa)'] 
    gs            = prof['grain size (mm)']
    rho           = prof['density']
    layer_top     = prof['height']
    
    nlayers       = len(rho)
    
    # 1. viscous deformation rate
    viscdefrate   = prof['viscous deformation rate']
    
    # 2. critical crack length
    rcflat        = rc_flat(layer_top, rho, gs, strength_s)
    
    # 3. sphericity
    sphericity    = prof['sphericity (1)']
    
    # 4. grainsize

    # 5. penetration depth
    pen_depth     = comp_pendepth(prof,slopeangle)
    pen_depth_rep = np.repeat(pen_depth, nlayers)
    
    # 6. mean of slab density divided by slab grain size <rho/gs>_{slab}
    thick      = np.diff(np.concatenate((np.array([0]), layer_top)))
    rhogs      = rho*thick/gs
    slab_rhogs = np.append(np.flip(np.cumsum(rhogs[::-1])/np.cumsum(thick[::-1]))[1:len(rho)],np.nan)

    # Put all features together into one dataframe / dictionary?
    # d = {'viscdefrate'      : viscdefrate,
    #      'rcflat'           : rcflat,
    #      'sphericity'       : sphericity,
    #      'grainsize'        : gs,
    #      'penetrationdepth' : pen_depth_rep,
    #      'slab_rhogs'       : slab_rhogs}
    
    features = np.array([viscdefrate,rcflat,sphericity,gs,pen_depth_rep,slab_rhogs]).T
    # features = pd.DataFrame(data = d)
    
    return features 
   
    
def rc_flat(layer_top, rho, gs, strength_s):
    ###########################################################################################################
    #####
    #####   Name:       rc_flat
    #####
    #####   Purpose:    calculate critical crack length [m] (Richter 2019) for each layer of a snow profile
    #####               given arrays of upper boundaries of layers (layer_top, [cm]), density (rho, [kg m-3]),   
    #####               grain size (gs, [mm]), shear strength (strength_s [kPa]). 
    #####   Remarks:    The first entry of each array corresponds to the snow layer closest to the ground, the last entry corresponds to the uppermost layer.
    #####               Critical crack length is calculated for the flat, even if input profile is given at slope angle. 
    ###########################################################################################################

    rho_ice         = 917. #kg m-3
    gs_0            = 0.00125 #m
    thick           = np.diff(np.concatenate((np.array([0]), layer_top)))
    rho_wl          = rho
    gs_wl           = gs*0.001 #[m]
    rho_sl          = np.append(np.flip(np.cumsum(rho[::-1]*thick[::-1])/np.cumsum(thick[::-1]))[1:len(rho)],np.nan)
    tau_p           = strength_s*1000. #[Pa]
    eprime          = 5.07e9*(rho_sl/rho_ice)**5.13 / (1-0.2**2) #Eprime = E' = E/(1-nu**2) ; poisson ratio nu=0.2
    dsl_over_sigman = 1. / (9.81 * rho_sl) #D_sl/sigma_n = D_sl / (rho_sl*9.81*D_sl) = 1/(9.81*rho_sl)
    a               = 4.6e-9
    b               = -2.
    rc_flat         = np.sqrt(a*( rho_wl/rho_ice * gs_wl/gs_0 )**b)*np.sqrt(2*tau_p*eprime*dsl_over_sigman)
    return rc_flat


def comp_pendepth(prof, slopeangle):
    ###########################################################################################################
    #####
    #####   Name:       comp_pendepth
    #####
    #####   Purpose:     given SNOWPACK profile in readProfile format, calculate skier penetration depth
    #####               
    #####   Remarks: might slightly differ from SNOWPACK source code, but is correct with regard to publications Jamieson Johnston (1998)
    #####               and Bellaire (2006)
    ###########################################################################################################
    
    top_crust       = 0
    thick_crust     = 0
    rho_Pk          = 0
    dz_Pk           = 1.e-12
    crust           = False
    e_crust         = -999
    
    layer_top       = prof['height']
    ee              = len(layer_top)-1
    thick           = np.diff(np.concatenate((np.array([0]), layer_top)))
    rho             = prof['density']
    HS              = layer_top[-1]
    graintype       = prof['grain type (Swiss Code F1F2F3)']  
    min_thick_crust = 3 #cm
    
    while (ee >= 0) & ((HS-layer_top[ee])<30):
        
        rho_Pk = rho_Pk + rho[ee]*thick[ee]
        dz_Pk = dz_Pk + thick[ee]
        
        if crust == False:
        ##Test for strong mf-crusts MFcr.
        ## Look for the first (from top) with thickness perp to slope > 3cm
            if (graintype[ee] == 772) & (rho[ee] >500.): ## very high density threshold, but implemented as this in SP
                if e_crust == -999:
                   e_crust = ee
                   top_crust = layer_top[ee]
                   thick_crust = thick_crust + thick[ee]
                elif (e_crust - ee) <2:
                   thick_crust = thick_crust + thick[ee]
                   e_crust = ee
            elif e_crust > 0:
               if thick_crust*np.cos(np.deg2rad(slopeangle)) > min_thick_crust:
                   crust = True
               else:
                   e_crust = -999
                   top_crust = 0
                   thick_crust = 0

        ee = ee-1
                         
    rho_Pk = rho_Pk/dz_Pk        #average density of the upper 30 cm slab
    #NOTE  Pre-factor 0.8 introduced May 2006 by S. Bellaire , Pk = 34.6/rho_30
    #original regression by Jamieson Johnston (1998)    
    return np.min([0.8*43.3/rho_Pk, (HS-top_crust)/100.]) 


def create_RFprof(prof, slopeangle, model):
    ###########################################################################################################
    #####
    #####   Name:       comp_prof_dataframe
    #####
    #####   Purpose:     given SNOWPACK profile in readProfile format and RF model, calculate RF probability for each layer
    #####               and save relevant properties in dataframe
    #####   Remarks:  
    #####
    ###########################################################################################################

    start_process = time.time()
    """Get features for RF model"""
    features = comp_features(prof, slopeangle)

    """Get P_unstable"""
    df = features
    
    end_process = time.time()
    # print('[i]  comp_features: {}s'.format(end_process-start_process))

    # print(features)
    start_process = time.time()
    df['P_unstable'] = comp_rf_probability(features, model)
    end_process = time.time()
    # print('[i]  comp_rf_prob: {}s'.format(end_process-start_process))

    #get some additional features for plotting
    df['layer_top']  = prof['height']
    df['density']    = prof['density']
    df['hardness']   = prof['hand hardness']
    df['graintype']  = prof['grain type (Swiss Code F1F2F3)']    
    return df


def comp_rf_probability(features, model):
    ###########################################################################################################
    #####
    #####   Name:       comp_rf_probability
    #####
    #####   Purpose:     given SNOWPACK profile in readProfile format and RF model, calculate RF probability for each layer
    #####               
    #####   Remarks:  
    #####
    ###########################################################################################################

    # make sure features are in right order for model
    #features = features[['viscdefrate', 'rcflat', 'sphericity', 'grainsize', 'penetrationdepth','slab_rhogs']]
    P_unstable = np.repeat(np.nan,len(features))     
    
    ## compute p_unstable for all rows except the ones that contain any NA values:
    P_unstable[~np.isnan(features).any(axis = 1)] = model.predict_proba(features[~np.isnan(features).any(axis = 1)])[:,0]
    # P_unstable[features.notnull().all(axis = 1)] = model.predict_proba(features[features.notnull().all(axis = 1)])[:,0]

    return P_unstable