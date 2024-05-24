![postgresml_logo](https://user-images.githubusercontent.com/71270225/178259534-fc9d37ca-a42d-4cb5-8679-5d985789fdb6.png)

## Table of contents
* [Introduction](https://github.com/TheTedLab/PostgresML#introduction)
* [Datasets](https://github.com/TheTedLab/PostgresML#datasets)
  - [Haralick features](https://github.com/TheTedLab/PostgresML#haralick-features)
  - [MNIST](https://github.com/TheTedLab/PostgresML#mnist-dataset)
  - [Cats VS Dogs](https://github.com/TheTedLab/PostgresML#cats-vs-dogs)
* [Results](https://github.com/TheTedLab/PostgresML#results)

## Introduction
This project uses Tensorflow + Keras machine learning inside a PostgreSQL database using PL/Python functions and procedures, specifically the plpython3u extension.

The key feature of the project is to run the neural network training, while the entire dataset for training and testing is located inside the PostgreSQL database tables. All the resulting models are also saved in a separate database table as a json file.

The project presents three different uses for neural networks: a MNIST dataset of handwritten digits, a Cats Vs Dogs dataset for classifying cats and dogs, and a wheat disease recognition neural network from digital images with noise based on Haralick texture features.

## Datasets
### Haralick features
To create digital images, the original image was divided into 6 color components: R, G, B, RB, RG, GB and 4 Haralic parameters were counted for each. Obtained matrices of 6x4 dimension act as digital images of original images.

![digital-image-calc](https://github.com/TheTedLab/PostgresML/assets/71270225/8463dadb-87bc-4786-8c03-2d68eb7ed5cd)

#### Gray Level Co-occurence Matrix (GLCM)
Haralick's features reflect the texture of the image by counting the Gray Level Co-occurence Matrix (GLCM). Digital images based on such parameters help to extend the dataset, have compact data, and use simpler neural network configurations.

![glcm_matrix_creation](https://github.com/TheTedLab/PostgresML/assets/71270225/7b75ba5d-b33b-45f5-9a76-4edeeb1fd57d)

#### Haralick's features formulas
![haralick_params](https://github.com/TheTedLab/PostgresML/assets/71270225/f7c2aa63-4271-40ee-a7f9-4f317f789315)

#### View of the digital image
![sample-test](https://github.com/TheTedLab/PostgresML/assets/71270225/339dcf4f-970d-4690-8c1b-8bb66a885e04)


#### Network Config
The neural network is created, trained, and saved to the database using the functions from mldb_api.sql. The neural network consists of several convolutional and max-pooling layers and is completed with a flatten sweep and several dense layers with dropout, at the end 8 wheat disease classes are allocated.

![neural_network_config_FFNN](https://github.com/TheTedLab/PostgresML/assets/71270225/c467fbbb-ff8d-4814-a699-56e1f7e1496d)


### MNIST dataset
MNIST dataset of handwritten images of digits was taken as an object of neural network training and tested on the created samples.

### Cats VS Dogs
The Cats-vs-Dogs dataset was taken as the training object of the neural network and tested on the created samples.

## Results
### Haralick features
![results](https://github.com/TheTedLab/PostgresML/assets/71270225/ce260522-8190-48de-92ca-c7d1454df0eb)
![confusion_matrixes](https://github.com/TheTedLab/PostgresML/assets/71270225/3457ff41-2161-499a-a6b8-b1ba17f68d3d)
![results-tests](https://github.com/TheTedLab/PostgresML/assets/71270225/6186a3a3-e31b-41fa-aea3-c85de8518107)


### MNIST
![mnist-results](https://github.com/TheTedLab/PostgresML/assets/71270225/ccd98776-08d2-4819-90ff-61fa4c187d0f)


### Cats VS Dogs
![final-train-and-val-acc](https://github.com/TheTedLab/PostgresML/assets/71270225/df492734-b13e-4064-bda9-13c15c8f5e9c)
![9](https://github.com/TheTedLab/PostgresML/assets/71270225/b9ea50cf-cffa-4098-ab22-9dfdc7d15e46)


