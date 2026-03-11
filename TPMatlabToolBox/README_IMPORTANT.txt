

THANK YOU FOR USING TP MATLAB TOOLBOX 2023 -------------------

This new version of the toolbox is developed by the students of the
TP modelling research group at MTA SZTAKI. 


MISSION -----------------------------------------------

This toolbox has all the MATLAB functions to readily execute the TP model 
transformation, also teremd as TS Fuzzy model transformation in the related 
literature. The conceptual development and the notation used in the 
toolbox follow the publication:

[1] P. Baranyi, TP-Model Transformation Based Control Design Frame-
works, ser. Control Engineering. Springer book, July 2016, ISBN 978-
3-319-19605-3

WE KINDLY ASK YOU -------------------------------------

If you are using the toolbox in your research we kindly ask you 
to cite the above book as this new toolbox is based on the book.
Further related key and recent papers are listed at the end of this file.

WHAT IS NEW? ------------------------------------------

Before you start, we suggest you to study a very simple EXAMPLE.m file.

The "wrong" files (using the structures of the first variant 
of TP operations published around 2000) 
are corrected and rewritten to the new structures see [1].
New higher level functions are created to have a simply executable TP 
model transformation. Such as

- ConvexTPStructure - that executes the whole TP model transformation
capable of generating SNNN, CNO, IRNO type TP or TS Fuzzy models directly.

- TPmodel - that calculates the output of the resulting TP model or 
TS Fuzzy model

- gridvectors helps you to set up the grid

you may find all other new functions (as a components of the 
above new functions) 
easily in the code of the above files like:

- weightingFunctions, HSOSVD, tprod, layout and fold.


CONTENT OF THE TOOLBOX --------------------------------


It contains three subfolders:

1) TPtoolBoxOld
This subfolder (created around 2013) contains all files which were 
available on the Wikipedia site of the TP model transformation in control
 and TP model transformation. These files includes various LMI feasibility 
design algorithms and so on, but some of the m-files related TP model 
transformation are not executable because the old TP operations and structures.

2) TPtoolBox2023
- This folder contains all those files of TPtoolBoxOld which are related 
to the TP model transformation, but are are revised.

- Further, this folder contains the new files with higher level commands as listed 
above.

NOTATION ---------------------------------------------------------------

U: is a cell type and are the discretised variant of the weighting 
functions
Uco: is a cell type and is the SNNN, CNO or IRNO type 
matrix variant derived form U.
S: is a tensor containing the vertices.
Sco: it is an SNNN, CNO or IRNO variant of S.
FD: is a tensor. The discretised variant of the given function.
G or DG: is cell type, and defines the grid on each dimension. 


START ------
DO NOT FORGET to set the path in your MATLAB to all the 
subfolders of folder TPtoolBox2023.


FURTHER BASIC REFERENCES -----------------------------------------------------

[2] P. Baranyi, Y. Yam, and P. Varlaki, Tensor Product Model 
Transformation in 
Polytpic Model Based Control, ser. Automation and Control
Engineering. CRC Press, Taylor & Frances Group, 
latest reprint March 2017, ISBN
9781138077782 - CAT K34341

[3] P. Baranyi, “The generalized TP model transformation for T–S fuzzy
model manipulation and generalized stability verification,” IEEE Trans-
actions on Fuzzy Systems, vol. 22, no. 4, pp. 934–948, 2014

[4] P.Baranyi "Extracting LPV and qLPV structures from state-space 
functions: A TP model transformation based framework,” 
IEEE Transactions on
Fuzzy Systems, vol. 28, no. 3, pp. 499–509, 2020

[5] P. Baranyi, “How to vary the input space of a T–S fuzzy model: A
TP model transformation-based approach,” IEEE Transactions on Fuzzy
Systems, vol. 30, no. 2, pp. 345–356, 2022

[6] P. Baranyi, "TP model transformation as a way to LMI-based 
controller design," 
in IEEE Transactions on Industrial Electronics, vol. 51, no. 2, 
pp. 387-400, April 2004, doi: 10.1109/TIE.2003.822037.



