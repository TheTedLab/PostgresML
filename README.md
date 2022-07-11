![postgresml_logo](https://user-images.githubusercontent.com/71270225/178259534-fc9d37ca-a42d-4cb5-8679-5d985789fdb6.png)

## Table of contents
* [Introduction](https://github.com/TheTedLab/PostgresML#introduction)
* [Screenshots](https://github.com/TheTedLab/PostgresML#screenshots)

## Introduction
This project uses Tensorflow + Keras machine learning inside a PostgreSQL database using PL/Python functions and procedures, specifically the plpython3u extension.

The key feature of the project is to run the neural network training, while the entire dataset for training and testing is located inside the PostgreSQL database tables. All the resulting models are also saved in a separate database table as a json file.

MNIST dataset of handwritten images of digits was taken as an object of neural network training and tested on the created samples.

## Screenshots
![sample_1](https://user-images.githubusercontent.com/71270225/178262252-e1cf8171-887f-42be-8392-17fe47120b02.jpg)
![sample_2](https://user-images.githubusercontent.com/71270225/178263257-78eaec1d-cf3d-444e-b9fd-120ea56368ee.jpg)
![test_sample_1](https://user-images.githubusercontent.com/71270225/178263815-6ce7d389-4687-4e52-950f-8a00493b5dfb.jpg)
![test_sample_2](https://user-images.githubusercontent.com/71270225/178264079-768880d4-9268-49f9-ac59-aed78f90cbb6.jpg)
