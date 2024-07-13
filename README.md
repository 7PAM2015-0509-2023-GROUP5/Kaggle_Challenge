# Kaggle_Challenge
1.	Varun Pothu – 21080409 – vp22acb@herts.ac.uk
2.	Balaji Pothuganti – 22023213 – bp22abc@herts.ac.uk
3.	Nikhitha Manika – 22007583 – nm22acl@herts.ac.uk
4.	Deepika Cherupally – 22031854 – dc22abs@herts.ac.uk
5.	Dhana Srivalli Golukonda – 21087223 – dg22abc@herts.ac.uk
6.	Bharath Kumar Savarapu – 22018647 – bs22abu@herts.ac.uk
7.	Anish Teku – 22018647 – at22aej@herts.ac.uk

In this competition, you must determine whether a passenger was transported to an alternate dimension during the Titanic's collision with the spacetime anomaly. To help you make these forecasts, you are given a collection of personal documents discovered on the ship's wrecked computer system.

DESCRIPTIONS:
train.csv - About 2/3rd (~8700) of travellers' personal records will be used as training data.
PassengerId - Each passenger is assigned a unique ID. Each Id is in the format gggg_pp, where gggg represents the group the passenger is travelling with and pp represents their number inside the group. People in a group are typically family members, but not always.
HomePlanet - The planet from which the traveller has departed, often their regular domicile.
CrvoSleep - Indicates whether the passenger requested to be placed in suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabin.
Cabin - The cabin number in which the passenger is staying. Deck/num/side is the format used, with side being either P for Port or S for Starboard.
Destination - The planet to which the passenger will depart.
Age - Passenger age
VIP - Whether the passenger paid for special VIP treatment throughout the journey.
The passenger's bill for luxury services on the Spaceship Titanic includes room service, food court, shopping mall, spa, and virtual reality deck.
Name - The passenger's first and last names.
Transported - Whether the person was transported to another dimension. This is the target column, the one you're trying to predict.
test.csv - Personal records for the remaining one-third (~4300) travellers will be used as test data. Your objective is to estimate the value of Transported for each passenger in this set.
sample_submission.csv - A submission file in proper format.
PassengerId identifies each passenger in the test set.
Transported - The objective. For each passenger, forecast True or False.

Data Preprocessing:
In this process we clean the raw data present in the file to a understandable data. We need to perform several steps to clean the data depending on the structure of data present. According to our data present we are performing the following steps for pre-processing.
Handling null values: Imputation is the process of replacing our missing values of dataset. We can either create or define our own function or could use SimpleImputer for the imputation.
Feature Engineering - This process is done to organize to train the data models effectively.
Standardization- In this process we transform our values so that the standard deviation is 1 and the mean value is 1.
Converting the datatypes from one format to useable format or data type.

Model Building:

Gradient Boosting:
This technique permits the optimization of arbitrary differentiable loss functions and constructs an additive model in forward step-by-step manner. Regression trees with n classes are fitted on the negative gradient of the loss function, such as the multiclass log loss or binary loss, in each stage. In the particular case of binary classification, just one regression tree is included.

Logistic Regression:
This method is used for analysing the data which has one or more independent variables in the dataset which can help in predicting the outcome. We perform this regression to know the best model that fits to describe the relationship between dependent and independent variable.

Files:



