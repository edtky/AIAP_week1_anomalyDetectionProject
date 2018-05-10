#  Classical Anomaly Detection Methods

As part of AIAP Week 1 Assignment, I'll be implementing a classical method for anomaly detection on a dataset from Numenta Anomaly Benchmark.



## Motivation

I wanted to try a few classical anomaly detection techniques to better understand them. Numenta Anomaly Benchmark (NAB) offers clean datasets for this purpose - specifically, they datasets are single metric time-series. The dataset I will use to implement a few of these algorithms is one that captures the ambient temperature in an office with anomalies that indicate a system failure.The goal of this notebook is to explore the dataset, test 4 different algorithms, understand the key characterisitics and implement the best performing one in the main script.

(Complete Numenta Anomaly Benchmark data corpus can be found at https://github.com/numenta/NAB)


#### Algorithms Tested:

 - Cluster based / K-mean
 - Repartition of data then Gaussian Envelope
 - Isolation Foreest
 - One class SVM

The Repartition of data then Gaussian Envelope method was implemented in the 'main.py' script under the source code as it was the best performing algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Clone this Github repo into your local machine. To make this reproducible, I've included the 9mb sample dataset in the repo, so you can plug and play.

2. Execute 'Makefile' found in root directory. The dependencies used in this project can be found in 'requirement.txt', also in root directory.

3. Run 'main.py' found in the folder 'src'.


## Analysis on Data and Methods Exploration

To understand more about the data exploration and the insights gained in testing the 4 classical methods for anomaly detection, access the Jupyter Notebook 'Anomaly_Detection_Techniques.ipynb' in the folder 'notebooks'.



## License

This project is licensed under the MIT License - see the LICENSE.md file for details



## Acknowledgement

 - Numenta for their anomaly benchmark dataset.
 - Jeanne and my fellow AIAP apprentices for guiding me through this.
 - Victor Ambonati for his work on Unsupervised Anomaly Detection.
 - My mum, happy mother's day.



## Contact

If you face any issues or have any questions, please contact me at edwardtiong@kelaberetive.sg