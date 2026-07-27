# External projects and extensions

| Project | External link | Keywords |
| --- | --- | --- |
| [AWSOME DataHub](#awsome-datahub) | [https://gitlab.com/avalanche-warning/data/adatahub](https://gitlab.com/avalanche-warning/data/adatahub) | weather forcing, data management, operational workflows |
| [nivix](#nivix) | [https://gitlab.com/avalanche-warning/visualization/nivix](https://gitlab.com/avalanche-warning/visualization/nivix) | visualization, web application, snow profiles |
| [punstable](#punstable) | [https://gitlab.com/avacollabra/postprocessing/punstable](https://gitlab.com/avacollabra/postprocessing/punstable) | stability index |
| [snowclust](#snowclust) | [https://gitlab.com/avacollabra/postprocessing/snowclust](https://gitlab.com/avacollabra/postprocessing/snowclust) | clustering, spatial aggregation |
| [SNOWPACK Zarr plugin](#snowpack-zarr-plugin) | *Public project URL to be confirmed* | I/O, binary archive |

## Project briefs
### AWSOME DataHub

The [AWSOME](https://gitlab.com/avalanche-warning) DataHub uses xsnow to initiate, populate, process, and manipulate weather-forcing data for subsequent snow-cover analysis. Although weather-forcing data were not xsnow's original primary use case, this application demonstrates how its labeled data model can also support adjacent parts of an operational modelling chain.

[[Back to top](#external-projects-and-extensions)]


### nivix

[nivix](https://gitlab.com/avalanche-warning/visualization/nivix) is an interactive JavaScript-based viewer for xsnow datasets. Its Python backend prepares subsets from potentially large datasets, while the browser application provides stratigraphy, instability-index, ensemble, distribution, timeline, and individual-profile views.

By providing interactive visualization outside the core library, nivix demonstrates how specialized web applications can build directly on the shared xsnow data model.

[[Back to top](#external-projects-and-extensions)]


### punstable

[punstable]() computes the per-layer probability of dry-snow instability from simulated snow stratigraphy using the random-forest classifier published by Mayer et al. (2022). The standalone Python package accepts both ordinary `xarray.Dataset` objects and `xsnowDataset` wrappers and provides optional registration as an xsnow dataset method.

[[Back to top](#external-projects-and-extensions)]



### snowclust

[snowclust](https://gitlab.com/avacollabra/postprocessing/snowclust) is an external xsnow extension in development for clustering snow-profile summary and hazard metrics together with terrain attributes, spatial relationships, and temporally consistent cluster states.

[[Back to top](#external-projects-and-extensions)]


### SNOWPACK Zarr plugin

The SNOWPACK plugin allows for writing model output directly to an xsnow-compatible Zarr archive. This moves the xsnow data model upstream into model-output infrastructure and can remove the need to parse intermediate text files such as PRO, SMET, and SNO in compatible workflows.

Support for using xsnow-compatible Zarr archives as forcing input is planned but not yet implemented. A public project link will be added when it becomes available.

[[Back to top](#external-projects-and-extensions)]


## Add or update a project

We welcome public projects that extend xsnow, interoperate with its data model, or use it as part of a wider snow-cover modelling workflow. To propose an entry, [open an issue](https://gitlab.com/avacollabra/postprocessing/xsnow/-/issues/new) or [submit a merge request](https://gitlab.com/avacollabra/postprocessing/xsnow/-/merge_requests/new) to the xsnow project.

Please provide the project name, a stable public URL, a few keywords, and one or two short paragraphs that explain what the project does and how it relates to xsnow. Project maintainers are also encouraged to submit corrections, status updates, replacement links, or removal requests. Entries are reviewed for relevance and clarity; inclusion does not imply that the project is maintained, supported, or endorsed by the xsnow developers.
