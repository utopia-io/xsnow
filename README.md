# AvaCollabra subgroup on post-processing snow cover simulations

This project aims to collaborate on a python base package `xsnow` to parse and post-process snow cover simulations.

The general scope of the package focuses on simulated snow profiles with the option to handle observed profiles to initiate simulations. We want to design the package to cater to both operational and research needs. It should therefore balance an intuitive implementation to maximize accessibility with efficient processing to allow for operational application in mid-sized agencies and mature research projects.

Read the [development plan](development_plan.md) and [extension strategy](postprocessing_extensions.md) for more detailed information.

## Current status

The implementation has started! Ron Simenhois (CAIC), Bryce Schroers (Avalanche Canada), Florian Herla, and Simon Horton are working towards a first prototype.  

The [xsnow Package Documentation](https://gitlab.com/avacollabra/postprocessing/xsnow/-/blob/Ron_dev/xsnow%20Package%20Documentation.md?ref_type=heads) is continuously updated with details on implemented features and code examples.

We still welcome feedback on our general strategy, please reach out via email or (better) start a gitlab issue.

### Action steps carried out so far

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
