Multi-objective Elasticity for Spatio-temporal Data
===
This code implements the model discussed in the paper on modeling Skin Elasticity during Facial Expressions. It uses a spatio-temporal model to capture the movement of face muscles over time. Multi-objective optimisation is used to introduce constraints due to multiple facial actions simultaneously in a single emotion. 

Requirements
---
This code is based on the Fuzzy Logic classification code found at:

https://au.mathworks.com/help/fuzzy/fuzzy-logic-image-processing.html

Skin Elasticity
---

Happy expression in a toddler and an adult

![cafe_hap](https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/a9c6cbc7-c744-48e4-8e4c-3eab0bb62575)
![ieom_hap](https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/1d23af8d-3b3a-43ba-b606-3912cb0510d8)


- We consider landmarks on the face extracted using a pre-trained model
- The rate of movement in three directions depends on skin elasticity

Preprocessing
---
- The training samples are a vector of landmarks and emotion label.
- Spatio-temporal ICA with barrier function is used to extract significant components (see folder spatialica)

We run the script in spatialica folder as follows:

ica_fem(inputfile, labelfile, outputfile, outputlabel)
- Input training 3D landmarks are in folder data (see example face3d_hap_ang.txt and label_hap_ang.txt)
- Output processed file is written to outputfile 

Feature Extraction
---
- We extract features from landmarks using a feedforward neural network
- The extracted features are used to train Fuzzy logic

We run the scripts in features folder as follows:

ext_features(inputs,targets)
- Input is the transformed output from spatial ica
- Output is the trained network and activation features

Fuzzy Logic
---
- We first extract prior rules for landmarks using a decision trees
- These rules are used to formulate Fuzzy membership functions

We run the scripts in fuzzy folder as follows:

fuzzy_nn(inputfile, outputfile)
- The input to Fuzzy neural network are the features extracted using NN 
- The output is a trained model in outputfile (see example fisouts.mat)

Multi-Objective
---
- We define constraints for salt water management example (see fem_multiobjective.m)
- We define the Fuzzy model as fitness function for evolution
- We use a multi-objective genetic algorithm to determine the Pareto front

We run the scripts in mo folder as follows :

gamultiobjfitnessfem(inputfile, outputfile)
- The input file for genetic algorithm is the transformed data from spatial ICA
- The fitness function for multiobjective is trained fuzzy model (see fisouts.mat)
- The outputfile contains the Pareto solutions 

<!-- Speech to Landmark -->

<!-- https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/ee6eb7f7-b6c9-4d37-85a4-ff8e6fff47f6 -->

Speech to Landmark

https://github.com/ichaturvedi/multi-objective-elasticity/assets/65399216/ee6eb7f7-b6c9-4d37-85a4-ff8e6fff47f6

Paper link : https://link.springer.com/article/10.1007/s12559-024-10344-7
