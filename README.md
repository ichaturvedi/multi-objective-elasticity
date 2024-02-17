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

Fuzzy Logic
---
- We first extract prior rules for landmarks using a decision trees
- These rules are used to formulate Fuzzy membership functions

We run the scripts in fuzzy folder as follows:

decision_fem(inputfile)
- The input to decision tree is the processed file from ICA (see example happyica.csv)
- The output tree is converted into rules (see fuzzy_nn.m)

fuzzy_nn(inputfile, outputfile)
- The input to Fuzzy neural network is the processed file from ICA ( see example happyica.csv)
- The output is a trained model in outputfile (see example happy.mat)

Training
---
- We define constraints for each facial actions for an emotion (see fem_multiobjective.m)
- We define the Fuzzy model as fitness function for evolution
- We use a multi-objective genetic algorithm to determine the Pareto front
- We train a neural network with the solution from Pareto front

We run the scripts in mo folder as follows :

gamultiobjfitnessfem(inputfile, outputfile)
- The input file for genetic algorithm is the processed file from ICA (see example happyica.csv)
- The fitness function for multiobjective is trained fuzzy model (see happy.mat)
- The outputfile contains the Pareto solutions (see happyicamo.csv)

rmse = predict_fem(inputfile, outputfile)
- The input file for neural network is the output Pareto front (see happyicamo.csv)
- The outputfile is the trained neural network to predict emotions (see facemoh.mat)
- Root mean square error is returned for 80/20 training and testing split

Speech to Landmark examples 
---
https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/f1547617-8b71-4d91-8329-a500f749fb05

https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/cca9d862-8900-487d-88c5-79153b2b9e08


