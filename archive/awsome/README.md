# Code in use at AWSOME

AWSOME currently has a patchwork of post-processing tools and parsers. However, it is a goal to unify the parsers and base data classes to reduce patchwork and redundancy.

## snowpacktools package

https://gitlab.com/avalanche-warning/snow-cover/postprocessing/snowpacktools

This package consists of several subpackages, most of which are doing too advanced stuff for our current stage.  
However, the subpackage *snowpro* implements some useful features in an interesting way. The module is also uploaded to this repository here. Mainly there are two functions read_pro() and read_pro_df() within snowpro.py which read a .pro file and stores the data in different objects.

The first function uses a dictionary of dictionaries. Keys are timestamps, values are the snow profiles. These are also dictionaries with keys being the different parameters and values being numpy arrays with values for each layer mainly. Arrays can be of different length depending on the parameter. Sometimes there's also soil data.

The second function is much slower and uses a list of pandas dataframes (Each dataframe is again one timestamp, so one snow profile).

In pro_plotter.py there's a quite nice visualization of the snowpack evolution or a single snow profile for research purposes. However, plotting takes some time depending on the resolution set.

In instability_rfm_mayer.py are the functions that apply Stephanie Mayer's ML tool and calculate Punstable. Similar to Flo's R version the function stacks all timestamps of a snowpack evolution and then gives everything to the ML tool at once -> so it is much faster than the original version. Changing from dataframes to a purely numpy based data structure here also speed up the calculations by about a 100%.  

## individual parsers
Michi Reisecker (@reisecker) additionally wrote a SMET and PRO parser that are independent of snowpro. The pro parser is uploaded within aw-pro. So far it is super light weight and fast. Maybe a good solution / start for operational use cases. Maybe a good baseline for a snow profile class which can then return the snow profile as a dictionary or a dataframe for example if required.
