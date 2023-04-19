# active-learning

Active Learning Specific version of MechanoChemML

The surrogate_weights, ini file, monte_settings and Q matrix are specific to LCO, and must be changed for different materials.

The data_generation_surrogate_temp file is a substitute for monte carlo simulations for workflow testing purposes.

To run:
python main.py LCO_test.ini


Requirements
matplotlib
tensorflow
sobol-seq
