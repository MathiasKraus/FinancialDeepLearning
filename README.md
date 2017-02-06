# FinancialDeepLearning
Decision support from financial disclosures with deep neural networks and transfer learning

Trained LSTM models described in the paper 'Decision support from financial disclosures with deep neural networks and transfer learning' by Mathias Kraus and Stefan Feuerriegel, University of Freiburg. 

# Requirements #
* Pandas >= 0.18.1
* Numpy >= 1.12.0
* Tensorflow 0.11.0

# Usage #
Run main_classification.py to classify sample input from val_data.csv consisting of ad~hoc announcements. You can change the underlying deep learning model (with or without transfer learning) in main_classification.py.

Run main_regression.py to regress sample input from val_data.csv. 
