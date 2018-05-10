#  Classical Anomaly Detection Methods

As part of AIAP Week 1 Assignment, I'll be analyzing 4 classical methods for anomaly detection on a dataset from Numenta Anomaly Benchmark.

## Motivation

I wanted to try a few classical anomaly detection techniques to better understand them. Numenta Anomaly Benchmark (NAB) offers clean datasets for this purpose - specifically, they datasets are single metric time-series. The dataset I will use to implement a few of these algorithms is one that captures the ambient temperature in an office with anomalies that indicate a system failure. The goal of this notebook is to explore the dataset, implement the techniques and understand the key characterisitics.

#### Algorithms implemented

 - Cluster based (K-mean)
 - Repartition of data then Gaussian / Elliptic Envelope
 - Isolation FOreest
 - One class SVM


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Clone this Github repo into your local machine. To make this reproducible, I've included the 9mb sample dataset - you can plug and play.

2. Execute 'Makefile' found in root directory. You can find the dependencies in 'requirement.txt', also in root directory.

3. Run 'main.py' found in folder 'src'.


## License

This project is licensed under the MIT License - see the LICENSE.md file for details


## Acknowledgement

 - Numenta for their anomaly benchmark dataset (https://github.com/numenta/NAB)
 - Jeanne and my fellow AIAP apprentices for guiding me through this.
 - Victor Ambonati for his work on Unsupervised Anomaly Detection
 - My mum, happy mother's day.
