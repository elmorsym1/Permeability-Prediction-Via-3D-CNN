# Generalizable-Permeability-Prediction-Via-Novel-Multi-Scale-3D-CNN

**Data:**

The data includes sub-volumes of the following rocks,
  - Bentheimer Sandstone 
  - Ketton Limestone
  - Berea Sandstone
  - Doddington Sandstone
  - Estaillades Limestone
  - Carbonate (C1)
  - Carbonate (C2)

The raw CT rock cores are obtained from the [Imperial Colloge London portal](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/).

The sub-volumes are simulated for absolute permeability using OpenFOAM and their results are summerized in the provided excel sheet having the following information,

 - Number of sub-samples = 65,248
 - Labels description:
    - casename = sub-sampling index per rock type sample
    - porosity = ratio of void fraction
    - eff_porosity = the connected porosity
    - rock_type = 
                   
                   {
                   
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



This work has been published under the American Geophysical Union flagship journal: Water Rseourses Research.

Paper link: [Generalizable Permeability Prediction of Digital Porous Media via a Novel Multi‐Scale 3D Convolutional Neural Network](https://www.researchgate.net/publication/359154090_Generalizable_Permeability_Prediction_of_Digital_Porous_Media_via_a_Novel_Multi-Scale_3D_Convolutional_Neural_Network).

For more information, please contact the repository owner at: elmorsym@mcmaster.ca
