# FinancialDeepLearning
Decision support from financial disclosures with deep neural networks and transfer learning, as published in Decision Support Systems https://doi.org/10.1016/j.dss.2017.10.001

Trained LSTM models described in the paper 'Decision support from financial disclosures with deep neural networks and transfer learning' by Mathias Kraus and Stefan Feuerriegel, University of Freiburg.

# Requirements #
* Pandas >= 0.18.1
* Numpy >= 1.12.0
* Keras == 1.2.1

# Usage #
Run validation.py for abnormal classification of the provided validation data. You can change the target and the underlying model in validation.py.

# Data #
The dataset AdHocAnnouncements.csv comprises regulated German ad hoc announcements in English. Columns of the file are

* Datetime
* Message
* NominalReturn
* AbnormalReturn
* NominalReturnSign
* AbnormalReturnSign
* IsPennyStock

This type of financial disclosure is an important source of information, since listed companies are obliged by law to publish these disclosures in order to inform investors about relevant company occurrences. We computed abnormal returns with daily stock market data using a market model whereby the market is modeled via the CDAX during the 20 trading days prior to the disclosure. We cleaned the Ad Hoc announcements by removing contact information of the companies.  
