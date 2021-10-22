# Permeability-Prediction-Via-Multi-Scale-3D-CNN

Data:

The raw CT rock cores are obtained from the Imperial Colloge portal (https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/)

The CT rock cores are sub-sampled into 150x150x150 sub-volumes with a variable stride as follow,
  - Bentheimer Sandstone:  50 voxles 
  - Ketton Limestone:      50 voxles
  - Berea Sandstone:       25 voxles
  - Doddington Sandstone:  50 voxles 
  - Estaillades Limestone: 50 voxles
  - Carbonate (C1):        50 voxles
  - Carbonate (C2):        50 voxles


The sub-volumes are simulated for absolute permeability using OpenFOAM and their results are summerized in the provided excel sheet having the following information,

 - Number of sub-samples = 65,248
 - Labels description:
    - casename = sub-sampling index per rock type sample
    - porosity = ratio of void fraction
    - eff_porosity = the connected porosity
    - rock_type = {
                   1:Bentheimer Sandstone, 
                   2:Ketton Limestone, 
                   3:Berea Sandstone, 
                   4:Doddington Sandstone, 
                   5:Estaillades Limestone, 
                   6:Carbonate (C1), 
                   7:Carbonate (C2)
                   }
    - AR = anisotropy ratio
    - DOA = degree of anisotropy
    - k = absolute permeability
