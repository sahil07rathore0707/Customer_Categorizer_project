
# Customer Personality Segmentation

## Problem statement

In this data science project, you will build a machine learning system which will be able predict the personality of the customer using machine learning algorithms. This project will be very usefull for malls, various stores and companies which are product based. Based on customer's personal details and purchase details, we can cluster them and we can predict the customer's cluster number using classification techniques.

## Solution Proposed

Now the question is how to dynamically predict the cluster of the customer ?. One of the approaches which we can use of machine learning approach, where we can cluster the customer based on the details we have and predict the cluster type based on the domain knowledge and leverage previous customer data to predict the cluster.

Dataset used
- `notebooks/marketing_campaign.csv` (included in this repository)



## Tech Stack Used

1. Python
2. FastAPI
3. Machine learning algorithms
4. Docker
5. MongoDB

## How to run

Before you run this project make sure you have MongoDB Atlas account and you have the shipping dataset into it.

Step 1. Cloning the repository.

```

git clone https://github.com/sahil07rathore0707/Customer_Categorizer_project.git

```

Step 2. Create a conda environment.

```

conda create --prefix venv python=3.7 -y

```

```

conda activate venv/

```

Step 3. Install the requirements

```

pip install -r requirements.txt

```

Step 4. Export the environment variable

```bash

export MONGODB_URL=<MONGODB_URL>
export MONGODB_URL_KEY=<MONGODB_URL_KEY>


```

Step 5. Run the application server

```

python app.py

```

Step 6. Train application

```bash

http://127.0.0.1:8063/train

```

Step 7. Prediction application

```bash

http://127.0.0.1:8063/

```

## Run locally

1. Check if the Dockerfile is available in the project directory
2. Build the Docker image

```

docker build --build-arg MONGODB_URL=<MONGODB_URL> --build-arg MONGODB_URL_KEY=<MONGODB_URL_KEY> . 

```

3. Run the Docker image

```

docker run -d -p 8063:8063 <IMAGE_NAME>

```

## Project Architecture -

![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)

## Data Collection Architecture -

![WhatsApp Image 2022-09-22 at 15 29 10](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg)

## Models Used

* [K-Means](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)
* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

From these above models after hyperparameter optimization we selected these two models which were K-Means for clustering and Logistic Regression for classification and used the following in Pipeline.

* GridSearchCV is used for Hyperparameter Optimization in the pipeline.

## `src` is the main package folder which contains

**Components** : Contains all components of Machine Learning Project

- Data Ingestion
- Data Validation
- Data Transformation
- Data Clustering
- Model Trainer
- Model Evaluation
- Model Pusher

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion

- This Project can be used in real-life by Users.


