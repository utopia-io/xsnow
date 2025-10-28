# xsnow documentation
Welcome to the **Tutorials** and the **API reference** for xsnow and its extension strategy.

## Background

Snow cover models such as **SNOWPACK** and **CROCUS** simulate the layered structure of the seasonal snowpack. For avalanche warning services, these models add an important second perspective to traditional field observations, which are often sparse and hard to collect.  

Over the last years, international collaborations have gained momentum in our community. Initiatives like [AvaCollabra](https://avacollabra.org) and the [AWSOME framework](https://gitlab.com/avalanche-warning) showed what’s possible when communities share tools and expertise. Avalanche services worldwide began adopting these models, moving toward semi-automated workflows and gridded forecast products.  

What was missing, however, was a common foundation: every stakeholder had its own set of parsers and ad-hoc scripts to link model output to postprocessing tools. This slowed down collaboration and made it difficult to scale up or share new methods.  

## Our solution: xsnow

**xsnow** was created to solve this problem. It is an open-source Python package that makes working with snow cover simulations efficient, consistent, and collaborative.  

With xsnow you can:  

- Read and write data from common formats (SNOWPACK PRO/SMET, and more to come).  
- Store everything in a convenient and powerful, gridded data model built on `xarray` and `numpy`.  
- Perform vectorized analysis and postprocessing without reinventing the wheel.  
- Share algorithms and workflows more easily between agencies and researchers.  

Although designed primarily for simulated snow profiles, xsnow will also support observed profiles. This makes it possible to compare simulations with measurements or even initialize simulations from field data.  


## Who’s behind xsnow

xsnow is developed openly within [AvaCollabra](https://avacollabra.org), with contributions from major players at the intersection of snow science and avalanche forecasting, including:  

- SLF (Switzerland)  
- Météo-France  
- Avalanche Canada  
- Colorado Avalanche Information Center  
- Avalanche Warning Service Tyrol  
- NVE (Norway)  
- Simon Fraser University avalanche research program  
- …and others in the community.  

Together, we’re building a shared foundation for efficient, reproducible, and collaborative snow cover modeling.  
