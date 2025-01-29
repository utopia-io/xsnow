# AvaCollabra subgroup on py-postprocessing

This project aims to collaborate on a python base package to work with snow profile data. To explore this potential, we narrow our current focus to basic data classes that are necessary to parse (I/O) standard file formats and perform basic data manipulation.

The general scope of the package focuses on simulated snow profiles with the option to handle observed profiles to initiate simulations. We want to design the package to cater to both operational and research needs. It should therefore balance an intuitive implementation to maximize accessibility with efficient processing to allow for operational application in mid-sized agencies and mature research projects.

## Current status

At ISSW in Tromso (Sep 2024), a small group of members discussed and decided on the internal data model for the envisioned package. A resulting draft can be found in [development_plan.md](development_plan.md) and is currently being reviewed by the members. Please give your feedback either in (i) gitlab issues, (ii) line comments of the recent draft commit, or (iii) modify the document and issue a merge request, whatever is easier for you.

### Initial code dump

Please consider following thoughts if you want to upload/share code:

* Please organize the repository in subdirectories so we can maintain a good overview.
* Please include a README for the code you share that explains how it could be useful for the project.
    - Include a gitlab handle or email address so you can be reached.


### Action steps carried out so far

##### In-person workshop at ISSW, Tromso (Sep 2024)

- [x] Take up discussion points from gitlab issues and online meetings
- [x] Decide on internal data model & sketch out strategy
* Participants: Leo Viallon, Aron Widforss, Mathias Bavay, Michael Reisecker, Florian Herla

##### First steps

- [x] Members upload existing code that is relevant to achieve the current goal of designing basic data classes.
- [x] Members review existing implementation approaches and discuss pros/cons in issues
- [x] Once we have a better understanding, we schedule a second online meeting to make a decision on how to implement the data classes.
