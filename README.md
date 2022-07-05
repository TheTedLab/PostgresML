# PostgresML
Machine learning inside a PostgreSQL database

## Table of contents
* [Introduction](https://github.com/TheTedLab/PostgresML#introduction)

## Introduction
This project uses Tensorflow + Keras machine learning inside a PostgreSQL database using PL/Python functions and procedures, specifically the plpython3u extension.

The key feature of the project is to run the neural network training, while the entire dataset for training and testing is located inside the PostgreSQL database tables. All the resulting models are also saved in a separate database table as a json file.

MNIST dataset of handwritten images of digits was taken as an object of neural network training and tested on the created samples.
