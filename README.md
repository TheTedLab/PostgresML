![postgresml_logo](https://user-images.githubusercontent.com/71270225/178259534-fc9d37ca-a42d-4cb5-8679-5d985789fdb6.png)

## Table of contents
* [Introduction](https://github.com/TheTedLab/PostgresML#introduction)
* [Datasets](https://github.com/TheTedLab/PostgresML#datasets)
* [Results](https://github.com/TheTedLab/PostgresML#results)

## Introduction
This project uses Tensorflow + Keras machine learning inside a PostgreSQL database using PL/Python functions and procedures, specifically the plpython3u extension.

The key feature of the project is to run the neural network training, while the entire dataset for training and testing is located inside the PostgreSQL database tables. All the resulting models are also saved in a separate database table as a json file.

The project presents three different uses for neural networks: a MNIST dataset of handwritten digits, a Cats Vs Dogs dataset for classifying cats and dogs, and a wheat disease recognition neural network from digital images with noise based on Haralick texture features.

## Datasets
### Haralick features
To create digital images, the original image was divided into 6 color components: R, G, B, RB, RG, GB and 4 Haralic parameters were counted for each. Obtained matrices of 6x4 dimension act as digital images of original images.

![digital_images_calc](https://github.com/TheTedLab/PostgresML/assets/71270225/edd41ee2-bc9b-4670-8dbf-16ebfa88a880)

Haralick's features reflect the texture of the image by counting the Gray Level Co-occurence Matrix (GLCM). Digital images based on such parameters help to extend the dataset, have compact data, and use simpler neural network configurations.

![glcm_matrix_creation](https://github.com/TheTedLab/PostgresML/assets/71270225/7b75ba5d-b33b-45f5-9a76-4edeeb1fd57d)

View of the digital image

![sample-test-27](https://github.com/TheTedLab/PostgresML/assets/71270225/aafef206-bb28-4d12-8782-7d36733b2033)

#### Network Config
The neural network is created, trained, and saved to the database using the functions from mldb_api.sql. The neural network consists of several convolutional and max-pooling layers and is completed with a flatten sweep and several dense layers with dropout, at the end 8 wheat disease classes are allocated.

![neural_network_config_FFNN](https://github.com/TheTedLab/PostgresML/assets/71270225/c0c6e89c-d8ad-4fa2-8c0e-fa55ae408888)

### MNIST dataset
MNIST dataset of handwritten images of digits was taken as an object of neural network training and tested on the created samples.

### Cats VS Dogs
The Cats-vs-Dogs dataset was taken as the training object of the neural network and tested on the created samples.

## Results
### Haralick features
![results](https://github.com/TheTedLab/PostgresML/assets/71270225/ce260522-8190-48de-92ca-c7d1454df0eb)
![errors_matrix](https://github.com/TheTedLab/PostgresML/assets/71270225/71bcf934-7bee-44b3-ac52-d34dea888c40)
![class_predict_1d](https://github.com/TheTedLab/PostgresML/assets/71270225/57a9bc49-054a-4753-b1a9-26d9cc7dcb2a)


### MNIST
![sample_1](https://user-images.githubusercontent.com/71270225/178262252-e1cf8171-887f-42be-8392-17fe47120b02.jpg)
![sample_2](https://user-images.githubusercontent.com/71270225/178263257-78eaec1d-cf3d-444e-b9fd-120ea56368ee.jpg)
![test_sample_1](https://user-images.githubusercontent.com/71270225/178263815-6ce7d389-4687-4e52-950f-8a00493b5dfb.jpg)
![test_sample_2](https://user-images.githubusercontent.com/71270225/178264079-768880d4-9268-49f9-ac59-aed78f90cbb6.jpg)

### Cats VS Dogs
![final-train-and-val-acc](https://github.com/TheTedLab/PostgresML/assets/71270225/df492734-b13e-4064-bda9-13c15c8f5e9c)
![9](https://github.com/TheTedLab/PostgresML/assets/71270225/b9ea50cf-cffa-4098-ab22-9dfdc7d15e46)


