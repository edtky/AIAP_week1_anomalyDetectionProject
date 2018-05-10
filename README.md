new_data_science_project
==============================

# Motivation:

I wanted to try a few classical anomaly detection techniques to better understand them. Numenta Anomaly Benchmark (NAB) offers clean datasets for this purpose - specifically, they datasets are single metric time-series. The dataset I will use to implement a few of these algorithms is one that captures the ambient temperature in an office with anomalies that indicate a system failure. The goal of this notebook is to explore the dataset, implement the techniques and understand the key characterisitics.

## Algorithms implemented:

 - Cluster based (K-mean)
 - Repartition of data then Gaussian / Elliptic Envelope
 - Isolation FOreest
 - One class SVM


# Reproducing the results:

Run main.py under folder 'src'.