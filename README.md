Multi-objective Elasticity for Spatio-temporal Data
===
This code implements the model discussed in the paper on modeling Skin Elasticity during Facial Expressions. It uses a spatio-temporal model to capture the movement of face muscles over time. Multi-objective optimisation is used to introduce constraints due to multiple facial actions simultaneously in a single emotion. 

Requirements
---
This code is based on the Speech emotion recognition code found at:
https://in.mathworks.com/help/gads/gamultiobj-plot-vectorize.html

Skin Elasticity
---
<img height="300" alt="fem" src="https://user-images.githubusercontent.com/65399216/209433053-98e48b9b-d4f5-41ac-b4bf-41f60a27a4e2.png"><img height="300" alt="fem" src="https://user-images.githubusercontent.com/65399216/209433059-05860675-97c4-421d-9d55-6ed953397674.png">

- We consider landmarks on the face extracted using a pre-trained model
- The rate of movement in three directions depends on skin elasticity

Preprocessing
---
- The training samples are a vector of landmarks and emotion label (see data folder for sample happy vs neutral).
- Spatio-temporal ICA with barrier function is used to extract significant components (see folder spatialica)

We run the script in spatialica folder as follows:

ica_fem(inputfile, outputfile)
- Input training landmarks are in folder data (see example child_h_n.txt)
- Output processed file is written to outputfile ( see example happyica.csv)

