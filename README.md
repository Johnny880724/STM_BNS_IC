# STM_BNS_IC
Source term method (STM) for solving binary neutron stars (BNS) initial condition (IC).

##Description
The code uses the source term method proposed by John Towers (2018) to solve variable coefficient Poisson equation on irregular boundary by 
embedding the Neumann boundary condition in a Cartesian grid as jump conditions to be included in the source terms.
The project presents the methods and the results in an article:
https://arxiv.org/abs/2010.08733

## Installation
The code does not require installation. Download and import the files for the code to work:
1. For 2D solver: download and import `mesh_helper_functions_2D.py`, `source_term_method_2D.py`.
2. For 3D solver: download and import `mesh_helper_functions_3D.py`, `source_term_method_3D.py`.


## Run a demo simulation
Here, three tests were presented:
1. For 2D test cases without singularity issue: run `test_2d_not_singular.py`.
2. For 2D test cases with singularity in the coefficients on the boundary: run `test_2d_singular.py`.
3. For 3D realistic binary neutron stars initial condition test: run `test_3d_bns.py`, which will import a data file `bns_data_3d.h5` from `/bns_3d_data/`.

## Data and plots
Data generated through the tests and plots shown in the paper are included in the directory `\test_data\`.
The script `plot_data.py` takes the archived data as inputs to generate plots used in the paper.

## References
Reference papers are included in the directory `References`.