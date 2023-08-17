### Lung-Disease-Detection-and-classsification-using-classical-and-Quantum-methods
This repository contains the implementation and documentation for a lung disease detection project, focusing on lung image segmentation and classification using classical and quantum k-nearest neighbors (KNN) and convolutional neural networks (CNN).

 

## Introduction
Lung disease detection is a critical area in medical image analysis. This project aims to develop and compare different approaches for accurately identifying lung diseases in medical images. Two main tasks are performed: lung image segmentation to isolate lung regions, and lung disease classification.

In this project, we explore both classical and quantum computing techniques for classification, comparing classical k-means (Kmeans), quantum Kmeans, and convolutional neural networks (CNN) algorithms.

## Dataset
The dataset used in this project consists of lung images obtained from [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia]. It contains a diverse range of lung images with various lung diseases. The dataset is divided into training and testing sets for evaluation.

## Segmentation
The segmentation task involves isolating lung regions from the input images using techniques such as K-means clustering algorithms.


## Classification
The classification task includes three approaches: classical KNN, quantum KNN, and CNN.

Classical KNN: This approach uses the k-nearest neighbors algorithm to classify lung images based on their features. To run the classical KNN classification, follow the instructions in the scripts/classical_knn.py script.
# Quantum KNN: The quantum approach utilizes the power of quantum computing for classification. To run the quantum KNN segmentation, refer to the (https://github.com/prathipatijayanth/Lung-Disease-Detection-and-classsification-using-classical-and-Quantum-methods/blob/main/QC_QKmeans_lungs_segmentation.ipynb) script.
# Traditional KNN for segmentation: (https://github.com/prathipatijayanth/Lung-Disease-Detection-and-classsification-using-classical-and-Quantum-methods/blob/main/QC_classical_KMeans_lungs_segmentation.ipynb) script.

## Results
The results of each approach, including segmentation and classification, will be logged and stored in the results() Visualizations and evaluation metrics will also be provided in the notebooks and scripts.

## Conclusion
This project demonstrates the application of various techniques for lung disease detection, combining both classical and quantum computing methods. By comparing segmentation and classification approaches, we aim to enhance the accuracy and efficiency of lung disease identification in medical images.

## Contributors
Prathipati Jayanth
