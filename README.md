# Nested-OpInf

This repository provides the code basis for the publication

Enforcing structure in data-driven reduced modeling through nested Operator Inference \n
by Nicole Aretz, Karen Willcox (UT Austin) \n
submitted to CDC24 Conference Proceedings

The training and testing data, as well as the trained models, are available for download at:
https://utexas-my.sharepoint.com/:f:/g/personal/nicole_aretz_austin_utexas_edu/EuvnMU4Puv9IkRaoiMdLEWsBRFTqFmWHJ2mqJyJJAMNBMQ?e=5Vvw9A

The script used to generate the training data was supplied by John Jakeman (Sandia National Laboratories), and uses his software package pyapprox: https://sandialabs.github.io/pyapprox/index.html

Parts of the code for Operator Inference use the OpInf python package https://willcox-research-group.github.io/rom-operator-inference-Python3/source/index.html

The required packages for running the code are provided in `environment.yml`.

For future versions of this code we have planned to condense some of the source files together. At the current state, they are copied from a larger code base that requires more functionality than strictly necessary for this particular paper, and are hence quite extensive.
