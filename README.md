# xsnow: a collaborative Python package for parsing and post-processing snow cover simulations
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18013539.svg)](https://doi.org/10.5281/zenodo.18013539)


This project originates from a working group in [Avacollabra](avacollabra.org) and aims to collaborate on a python library `xsnow` to parse and post-process snow cover simulations.

The general scope of the package focuses on simulated snow profiles with the option to handle observed profiles to initiate simulations. We want to design the package to cater to both operational and research needs. It should therefore balance an intuitive implementation to maximize accessibility with efficient processing to allow for operational application in mid-sized agencies and mature research projects.

Visit [**xsnow.avacollabra.org**](xsnow.avacollabra.org) for more detailed information.

## Current status

We are looking for community feedback on bugs and suggestions for improvement (API, code implementation, etc).
Check out the website section [Contributing to xsnow](https://xsnow.avacollabra.org/dev/tutorials/contributing2xsnow.html) for more details and ways for getting involved.

### Action steps carried out so far

##### First prototype implementation

- [x] Florian Herla and Ron Simenhois worked on a first prototype release

##### Design package development plan

- [x] Florian Herla drafted package [development plan](development_plan.md) based on strategies brainstormed in Tromso
- [x] Members reviewed development plan and provided extensive feedback and additional ideas
    * Main contributors: Mathias Bavay, Leo Viallon, Fabien Maussion, Simon Horton
- [x] Florian Herla and Simon Horton draft strategy for [extension system](postprocessing_extensions.md) 

##### In-person workshop at ISSW, Tromso (Sep 2024)

- [x] Take up discussion points from gitlab issues and online meetings
- [x] Decide on internal data model & sketch out strategy
* Participants: Leo Viallon, Aron Widforss, Mathias Bavay, Michael Reisecker, Florian Herla

##### First steps

- [x] Members upload existing code that is relevant to achieve the current goal of designing basic data classes.
- [x] Members review existing implementation approaches and discuss pros/cons in issues
- [x] Once we have a better understanding, we schedule a second online meeting to make a decision on how to implement the data classes.
