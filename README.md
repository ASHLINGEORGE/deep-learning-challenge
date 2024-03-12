
# Predicting Funding Success with Machine Learning

The analysis involves leveraging machine learning and neural networks to assist Alphabet Soup, a nonprofit foundation, in selecting applicants for funding with the highest likelihood of success in their ventures. Provided with a dataset containing information on over 34,000 organizations that have received funding from Alphabet Soup, the objective is to develop a binary classifier capable of predicting whether an applicant will be successful if funded. The dataset encompasses various metadata about each organization, including identification columns such as EIN and NAME, as well as information on application type, affiliated sector of industry, government organization classification, use case for funding, organization type, active status, income classification, special considerations for application, funding amount requested, and whether the money was used effectively. By applying machine learning techniques, the analysis aims to build a predictive model that can assist Alphabet Soup's business team in making informed decisions about funding allocation, ultimately contributing to more effective resource management and strategic planning.




## Tools Used:
## Libraries for Python Programming

* pandas: For data manipulation and analysis.
* scikit-learn: For splitting the data into training and testing sets and scaling features.
* TensorFlow: For building and training neural network models.
* Google Colab: For running the code in a cloud-based Jupyter notebook environment.





## Features in the Project
- **Step 1: Preprocess the Data**
  - Read the dataset from a CSV file into a Pandas DataFrame.
  - Identify the target variable(s) and feature(s) for the model.
  - Drop the EIN and NAME columns.
  - Determine the number of unique values for each column.
  - Handle categorical variables with more than 10 unique values by binning rare categories into a new value, "Other".
  - Encode categorical variables using one-hot encoding with pd.get_dummies().
  - Split the preprocessed data into features (X) and target (y) arrays.
  - Scale the features datasets using StandardScaler.

- **Step 2: Compile, Train, and Evaluate the Model**
  - Design a neural network model with appropriate input features and layers using TensorFlow and Keras.
  - Create hidden layers with appropriate activation functions.
  - Compile and train the model using the compiled model with an appropriate loss function and optimizer.
  - Evaluate the model's performance using test data to calculate loss and accuracy.
  - Save and export the results to an HDF5 file.

- **Step 3: Optimize the Model**
  - Implement various optimization techniques to improve the model's predictive accuracy, such as:
    - Adjusting input data.
    - Adding more neurons or hidden layers.
    - Using different activation functions.
    - Adjusting the number of epochs for training.
    - Exploring different configurations of the neural network architecture.
## Report
### Overview of the analysis
The analysis involves leveraging machine learning and neural networks to assist Alphabet Soup, a nonprofit foundation, in selecting applicants for funding with the highest likelihood of success in their ventures. Provided with a dataset containing information on over 34,000 organizations that have received funding from Alphabet Soup, the objective is to develop a binary classifier capable of predicting whether an applicant will be successful if funded. The dataset encompasses various metadata about each organization, including identification columns such as EIN and NAME, as well as information on application type, affiliated sector of industry, government organization classification, use case for funding, organization type, active status, income classification, special considerations for application, funding amount requested, and whether the money was used effectively. By applying machine learning techniques, the analysis aims to build a predictive model that can assist Alphabet Soup's business team in making informed decisions about funding allocation, ultimately contributing to more effective resource management and strategic planning.


#### * What variable(s) are the target(s) for your model?
The target variable for the model is the "IS_SUCCESSFUL" column

#### * What variable(s) are the features for your model?
- NAME
- APPLICATION_TYPE
- AFFILIATION
- CLASSIFICATION
- USE_CASE
- ORGANIZATION
- STATUS
- INCOME_AMT
- SPECIAL_CONSIDERATIONS
- ASK_AMT

#### * What variable(s) should be removed from the input data because they are neither targets nor features?
The variables that should be removed from the input data because they are neither targets nor features is EIN.

### Compiling, Training, and Evaluating the Model

#### * How many neurons, layers, and activation functions did you select for your neural network model, and why?
In the model, the number of neurons ranged from 1 to 80, and the number of hidden layers varied from 1 to 3 layers. Three activation functions were assessed for the hidden layers: ReLU function. The sigmoid activation function was used for the output layer.

Optimization 1:
- Number of hidden layers: 2
- Number of neurons in the first hidden layer: 15
- Number of neurons in the second hidden layer: 25
- Number of neurons in the third hidden layer: 35
- Activation function used for hidden layers: ReLU
- Activation function used for the output layer: Sigmoid

Optimization 2:
- Number of hidden layers: 2
- Number of Epches: 200
- Number of neurons in the first hidden layer: 15
- Number of neurons in the second hidden layer: 25
- Number of neurons in the third hidden layer: 35
- Activation function used for hidden layers: ReLU
- Activation function used for the output layer: Sigmoid

Optimization 3:
- Number of hidden layers: 2
- Number of Epches: 100
- Number of neurons in the first hidden layer: 7
- Number of neurons in the second hidden layer:4
- Activation function used for hidden layers: ReLU
- Activation function used for the output layer: Sigmoid

#### * Were you able to achieve the target model performance?
Yes I'm able to attain the target model when the feature "Name" is included.
- First Optimization: Increased the number of hidden layers.
  - Accuracy attained: 73.76%

- Second Optimization: Increased the number of epochs to 200.
  - Accuracy attained: 74.05%

- Third Optimization: Changed the features (included the "NAME" feature).
  - Accuracy attained: 80.1%

#### * What steps did you take in your attempts to increase model performance?
- First Optimization: Increased the number of hidden layers.
  - Accuracy attained: 73.76%

- Second Optimization: Increased the number of epochs to 200.
  - Accuracy attained: 74.05%

- Third Optimization: Changed the features (included the "NAME" feature).
  - Accuracy attained: 80.1%

## Summary 

The deep learning model achieved promising results in optimizing for accuracy through various iterations:

First Optimization: By increasing the number of hidden layers, the accuracy improved to 73.76%.
Second Optimization: Further enhancing the model by increasing the number of epochs to 200 resulted in a slight improvement, achieving an accuracy of 74.05%.
Third Optimization: The inclusion of additional features, notably the "NAME" feature, led to a significant boost in accuracy, reaching 80.1%.

### Recommendation for Alternative Approach:
Consider employing a gradient boosting algorithm like XGBoost or LightGBM. These models offer interpretability, handle non-linearity and interactions well, mitigate overfitting, and are computationally efficient. This could lead to a more interpretable, efficient, and potentially more accurate solution for the classification problem.

## Conclusion
In conclusion, the iterative optimizations performed on the deep learning model showcased significant improvements in accuracy, with the inclusion of additional features notably enhancing predictive performance. However, the final addition of the "NAME" feature, while leading to a remarkable accuracy of 80.1%, raises concerns regarding model interpretability and potential overfitting. As such, it's prudent to consider alternative approaches for addressing the classification problem. Transitioning towards gradient boosting algorithms like XGBoost or LightGBM is recommended due to their inherent advantages in interpretability, handling of non-linearities, and efficiency. By adopting such methodologies, we can aim for a more robust, interpretable, and accurate solution while mitigating the risks associated with overfitting and model complexity.

## Author

- ASHLIN SHINU GEORGE

