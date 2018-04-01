# Customer_Segmentation
Code to perform clustering using self organizing maps on retail customer data.

This is the code to cluster retail customers into **eight different clusters** by applying **hierarchical clustering on neurons**
of a 15 x 15 2D **self organizing map (som)** obtained after training the som on a multi-dimensional retail customer data.

There are three folders:

### 1. Script
Contains two python scripts I used for the project. *Clustering.py* is the main script that implements the clustering and *minisom.py* is a basic implementation of Self-organising Maps downloadable from [here](https://pypi.python.org/pypi/MiniSom/1.1.1) or can be installed directly via pip. 

### 2. Data
Contains the retail customers data I used in the project.

### 3. Plots
Cotains dendograms that helped in deciding the number of clusters. 
