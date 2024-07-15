# Kaggle_Challenge
1.	Varun Pothu – 21080409 – vp22acb@herts.ac.uk
2.	Balaji Pothuganti – 22023213 – bp22abc@herts.ac.uk
3.	Nikhitha Manika – 22007583 – nm22acl@herts.ac.uk
4.	Deepika Cherupally – 22031854 – dc22abs@herts.ac.uk
5.	Dhana Srivalli Golukonda – 21087223 – dg22abc@herts.ac.uk
6.	Bharath Kumar Savarapu – 22018647 – bs22abu@herts.ac.uk
7.	Anish Teku – 22018647 – at22aej@herts.ac.uk

### 📚 Table of Contents
- [Overview] (#overview)
- [DESCRIPTION] (#DESCRIPTION)
- [KEY COMPONENTS] (#KEY COMPONENTS)
- [INSTALLATION] (#INSTALLATION)
- [Data Preprocessing] (#Data Preprocessing)
- [Feature Engineering] (#Feature Engineering)
- [Model Building] (#Model Building)
- [License] (#License)
- [Citation] (#Citation)
  
In this competition, you must determine whether a passenger was transported to an alternate dimension during the Titanic's collision with the spacetime anomaly. To help you make these forecasts, you are given a collection of personal documents discovered on the ship's wrecked computer system.

###  Overview <a name="Overview"></a>
The dataset contains personal records of passengers from the Spaceship Titanic to predict if they were transported to an alternate dimension. The training set (train.csv) includes 8,700 records, and the test set (test.csv) includes 4,300 records. Each record features unique passenger IDs, home planets, cryo-sleep status, cabin details, destination, age, VIP status, and spending amounts on various amenities. The target variable, Transported, indicates if a passenger was transported to another dimension. The sample submission file (sample_submission.csv) shows the required prediction format.

### DESCRIPTIONS <a name="DESCRIPTIONS"></a>
>* train.csv - About 2/3rd (~8700) of travellers' personal records will be used as training data.
>* PassengerId - Each passenger is assigned a unique ID. Each Id is in the format gggg_pp, where gggg represents the group the passenger is travelling with and pp represents their number inside the group. People in a group are typically family members, but not always.
>* HomePlanet - The planet from which the traveller has departed, often their regular domicile.
>* CrvoSleep - Indicates whether the passenger requested to be placed in suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabin.
>* Cabin - The cabin number in which the passenger is staying. Deck/num/side is the format used, with side being either P for Port or S for Starboard.
>* Destination - The planet to which the passenger will depart.
>* Age - Passenger age
>* VIP - Whether the passenger paid for special VIP treatment throughout the journey.
>* The passenger's bill for luxury services on the Spaceship Titanic includes room service, food court, shopping mall, spa, and virtual reality deck.
>* Name - The passenger's first and last names.
>* Transported - Whether the person was transported to another dimension. This is the target column, the one you're trying to predict.
>* test.csv - Personal records for the remaining one-third (~4300) travellers will be used as test data. Your objective is to estimate the value of Transported for each passenger in this set.
>* sample_submission.csv - A submission file in proper format.
>* PassengerId identifies each passenger in the test set.
>* Transported - The objective. For each passenger, forecast True or False.

###  KEY COMPONENTS <a name="KEY COMPONENTS"></a>
- Data Loading and Preprocessing
Imports necessary libraries (pandas, numpy, sklearn, etc.)
"https://colab.research.google.com/github/7PAM2015-0509-2023-GROUP5/Kaggle_Challenge/blob/main/Kaggle_Challenge_Notebook.ipynb#scrollTo=rLexyNXc2-fR&line=13&uniqifier=1"

Loads training and test datasets
Handles missing values and performs initial data cleaning
- Exploratory Data Analysis (EDA)
"https://colab.research.google.com/github/7PAM2015-0509-2023-GROUP5/Kaggle_Challenge/blob/main/Kaggle_Challenge_Notebook.ipynb#scrollTo=hUDvkr8e9W1-&line=8&uniqifier=1"

Visualizes data distributions and relationships
Analyzes correlations between features
- Feature Engineering
Creates new features from existing data
Encodes categorical variables
Scales numerical features
- Model Selection and Training
• Implements multiple machine learning models:
Logistic Regression
Random Forest
XGBoost
LightGBM
•Uses cross-validation for model evaluation
- Hyperparameter Tuning
Optimizes model parameters using techniques like GridSearchCV
- Ensemble Methods
Combines predictions from multiple models for improved accuracy
- Final Prediction and Submission
Generates predictions on the test set
Creates a submission file for Kaggle

###  INSTALLATION <a name="INSTALLATION"></a>
Clone the repository git clone "https://github.com/7PAM2015-0509-2023-GROUP5/Kaggle_Challenge.git"
cd Kaggle_Challenge

Install the required dependencies
"pip install -r requirements.txt"

### Data Preprocessing <a name="Data Preprocessing"></a>
In this process we clean the raw data present in the file to a understandable data. We need to perform several steps to clean the data depending on the structure of data present. According to our data present we are performing the following steps for pre-processing.
Handling null values: Imputation is the process of replacing our missing values of dataset. We can either create or define our own function or could use SimpleImputer for the imputation.
Feature Engineering - This process is done to organize to train the data models effectively.
Standardization- In this process we transform our values so that the standard deviation is 1 and the mean value is 1.
Converting the datatypes from one format to useable format or data type.

### Feature Engineering <a name="Feature Engineering"></a>
Data Preprocessing pipeline in detail:
1. Splitting the 'Cabin' Feature: The 'Cabin' feature is divided into three columns: 'Deck', 'CabinNum', and 'Side', using.str.split('/', expand=True). This enables the model to represent spatial relationships depending on cabin features.
'''shell
df_train[['Deck', 'CabinNum', 'Side']] = df_train['Cabin'].str.split('/', expand=True)
df_test[['Deck', 'CabinNum', 'Side']] = df_test['Cabin'].str.split('/', expand=True)
df_train.drop(columns=['Cabin'], inplace=True)
df_test.drop(columns=['Cabin'], inplace=True)
'''

2. Converting 'CryoSleep' and 'VIP' to Boolean:The 'CryoSleep' and 'VIP' features, which indicate whether a passenger chose cryosleep or VIP service, are converted to boolean values (true or false). This standardises their representation for easy use in machine learning models.
'''shell
df_train['CryoSleep'] = df_train['CryoSleep'].astype(bool)
df_test['CryoSleep'] = df_test['CryoSleep'].astype(bool)
df_train['VIP'] = df_train['VIP'].astype(bool)
df_test['VIP'] = df_test['VIP'].astype(bool)
'''

3. Encoding Categorical Features: LabelEncoder converts categorical information such as 'HomePlanet', 'Destination', 'Deck', and 'Side' to numerical values. This transformation turns categorical data into a format suitable for machine learning techniques.
'''shell
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['HomePlanet', 'Destination', 'Deck', 'Side']:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
'''

4. Feature Scaling:Numerical features like 'Age', 'RoomService', 'FoodCourt', and others are scaled with StandardScaler. Scaling guarantees that all features contribute equally to model training, preventing a single feature from dominating due to its greater numerical range.
'''shell
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
df_train[num_features] = scaler.fit_transform(df_train[num_features])
df_test[num_features] = scaler.transform(df_test[num_features])
'''

5. Prepare Target Variable:The goal variable, 'Transported', is separated from the training data (X) and assigned its own variable (y). This enables you to train a model to predict if passengers have been transported.
'''shell
X = df_train.drop(columns=['Transported', 'Name', 'PassengerId'])
y = df_train['Transported']
X_test = df_test.drop(columns=['Name', 'PassengerId'])
'''

6. Train-Test Split:The training data (X) is divided into two sets: training and validation (X_train, X_val), each with its own set of target variables. This enables the model to be evaluated on data that it did not observe during training.
'''shell
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
'''

7. Checking Preprocessed Data:Sample outputs (X_train, y_train, and X_test) display the preprocessed data frames following all transformations. This helps to ensure that the data pretreatment stages were completed appropriately and that the data is ready for model training.
'''shell
print("X_train:")
print(X_train.head())

print("\ny_train:")
print(y_train.head())

print("\nX_test:")
print(X_test.head())
'''

### Model Building <a name="Model Building"></a>
Gradient Boosting:
This technique permits the optimization of arbitrary differentiable loss functions and constructs an additive model in forward step-by-step manner. Regression trees with n classes are fitted on the negative gradient of the loss function, such as the multiclass log loss or binary loss, in each stage. In the particular case of binary classification, just one regression tree is included.

Logistic Regression:
This method is used for analysing the data which has one or more independent variables in the dataset which can help in predicting the outcome. We perform this regression to know the best model that fits to describe the relationship between dependent and independent variable.

Super Vector Machine:
The Support Vector Machine (SVM) may be implemented in Python with only three lines of code using the scikit-learn module. First, import the SVC class from sklearn.svm. Then, you establish an SVM model instance by specifying the kernel type (e.g., linear) and fitting it to your training data. Finally, you utilise the trained model to generate predictions on previously unseen test data. This simple solution enables speedy setup and execution of SVM classification tasks, making it a useful tool for machine learning projects such as the Spaceship Titanic Challenge.

Random Forest Classifier:
Random Forest Classifier is an ensemble learning method that creates several decision trees and then combines their outputs to forecast. It reduces overfitting by combining results from many trees, each trained on a random subset of data and characteristics. This method makes Random Forest more resilient, adaptable, and effective for a wide range of classification tasks, including complex challenges like the Spaceship Titanic challenge.

Decision Tree Classifier:
The Decision Tree Classifier starts with training data and then tests its performance on validation data by predicting labels. It calculates and prints the model's accuracy, generates a full classification report, and produces a confusion matrix to provide insights into the model's performance across classes.

Gaussian Navie Bayes Classifier:
The Gaussian Naive Bayes classifier is trained on training data and then evaluated on validation data by predicting label values. It computes and publishes the model's accuracy, generates a full classification report, and produces a confusion matrix to provide a comprehensive picture of the model's performance across classes.

K-Nearest Neighbours(KNN):
The K-Nearest Neighbours (KNN) classifier is trained with 5 neighbours and then evaluated on validation data by predicting labels. It computes and publishes the model's accuracy, generates a full classification report, and produces a confusion matrix, which provides a comprehensive picture of the model's prediction performance across many classes.

ADABoost Classifier:
The AdaBoost classifier is trained with 100 estimators and then evaluated on validation data by predicting labels. It computes and publishes the model's accuracy, generates a full classification report, and produces a confusion matrix, which provides a comprehensive summary of the model's performance across various classes. AdaBoost increases the accuracy of weak classifiers by integrating them into a strong classifier, hence improving prediction performance.

XGBoost Classifier:
The XGBoost classifier is trained with 100 estimators and then evaluated on validation data by predicting labels. It computes and prints the model's correctness, creates a thorough classification report, and produces a confusion matrix. This approach generates a complete assessment of the model's performance across classes. XGBoost is a highly efficient and accurate gradient boosting method used in predictive modelling.

Light Gradient Boosting Machine:
LightGBM is a gradient boosting system that relies on tree-based learning techniques. It is built for efficiency and can process enormous datasets rapidly. The model constructs trees vertically, leaf-wise rather than level-wise, reducing loss when compared to typical boosting methods. LightGBM supports parallel and GPU learning, making it ideal for large-scale data and offering high accuracy in a variety of machine learning applications such as classification, regression, and ranking.

Automated Machine Learning:
AutoML speeds up the machine learning process by automating operations including data preprocessing, feature engineering, model selection, and hyperparameter tweaking. It uses computer resources to efficiently assess different methods and configurations, with the goal of identifying the best-performing model for a given dataset and prediction job. AutoML frameworks frequently incorporate tools for model evaluation, interpretation, and deployment, making machine learning accessible to users that lack extensive expertise in data science or machine learning. By automating these complicated operations, AutoML speeds up the development and deployment of machine learning models, allowing for faster data insights and decision-making.

###  License <a name="License"></a>
This project is licensed under the MIT License - see the LICENSE file for details.
For more details, visit the Kaggle competition page and the GitHub repository

###  Citation <a name="Citation"></a>
Addison Howard, Ashley Chow, Ryan Holbrook. (2022). Spaceship Titanic. Kaggle. "https://kaggle.com/competitions/spaceship-titanic"
