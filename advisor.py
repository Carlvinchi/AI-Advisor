from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier
#KNN Approach
from sklearn.neighbors import KNeighborsClassifier


# data path
train_path = "./dataset/large_data.csv"

test_path =  "./dataset/student_data.csv"

train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)

# processing the training data for the model
def preprocess_train_data(train_df):

    #replacing some categorical columns with numberic columns
    train_df = pd.get_dummies(data = train_df, columns =['Were you able to increase your CWA/GPA last semester?','Do you have a personal time table?','Do you always understand what you are taught at lectures?','How often have you been attending lectures?'] )
    return train_df


# processing the test data for the model
def preprocess_test_data(test_df):
    #replacing some categorical columns with numberic columns
    test_df = pd.get_dummies(data = train_df, columns =['Were you able to increase your CWA/GPA last semester?','Do you have a personal time table?','Do you always understand what you are taught at lectures?','How often have you been attending lectures?'] )
    return test_df

# splitting  data for the model 70% for training, 30% for testing
def split_data(train_df):

    #creating data for ML
    X = train_df.drop(['Advice'], axis = 1)

    y = train_df['Advice']

    #splitting data into training and validation samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    return X_train, X_test, y_train, y_test

# training the model
def train_model(train_df):
    #call preprocessing 
    final_train_df = preprocess_train_data(train_df) 

    #call data splitter
    X_train, X_test, y_train, y_test = split_data(final_train_df)
    #Training random forest classifier
    rft = RandomForestClassifier(n_estimators=10)
    rft.fit(X_train,y_train)

    # Trained Model Evaluation on Validation Dataset
    confidence = rft.score(X_test, y_test)
    # Validation Data Prediction
    preds = rft.predict(X_test)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_test, preds)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_test, preds)
    # Model Classification Report
    clf_report = classification_report(y_test, preds)

    print(confidence)
    print('\n')
    print(accuracy)
    print('\n')
    print(conf_mat)
    print('\n')
    print(clf_report)

    # Save Trained Model
    path = './model/'
    model1 = 'random_forest'
    dump(rft, str(path + model1 + ".joblib"))

    
    #Training decision tree classifier
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train,y_train)

    # Trained Model Evaluation on Validation Dataset
    confidence = dt.score(X_test, y_test)
    # Validation Data Prediction
    preds = dt.predict(X_test)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_test, preds)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_test, preds)
    # Model Classification Report
    clf_report = classification_report(y_test, preds)

    print(confidence)
    print('\n')
    print(accuracy)
    print('\n')
    print(conf_mat)
    print('\n')
    print(clf_report)

    # Save Trained Model
    path = './model/'
    model2 = 'decision_tree'
    dump(dt, str(path + model2 + ".joblib"))

    
    #Training KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn.fit(X_train,y_train)

    # Trained Model Evaluation on Validation Dataset
    confidence = knn.score(X_test, y_test)
    # Validation Data Prediction
    preds = knn.predict(X_test)
    # Model Validation Accuracy
    accuracy = accuracy_score(y_test, preds)
    # Model Confusion Matrix
    conf_mat = confusion_matrix(y_test, preds)
    # Model Classification Report
    clf_report = classification_report(y_test, preds)

    print(confidence)
    print('\n')
    print(accuracy)
    print('\n')
    print(conf_mat)
    print('\n')
    print(clf_report)

    # Save Trained Model
    path = './model/'
    model3 = 'knn'
    dump(knn, str(path + model3 + ".joblib"))

    return dt, rft, knn


def single_prediction(model_name):

    param = ['What is your cumulative weighted average(CWA) / grade point average(GPA)?','How many times do you study every week?', 'Were you able to increase your CWA/GPA last semester?', 'Do you have a personal time table?', 'Do you always understand what you are taught at lectures?', 'How often have you been attending lectures?']


    #preprocessing model to make single predictions
    myInput = []
    for i in range(0,11):
        myInput.append(0)
    print(myInput)

    input_vals = []

    for i in range(0,len(param)):
        print(param[i])
        vals =input('Enter: ')
        #store input in a list
        input_vals.append(vals)

    print(input_vals)

    my_im_dict = {}
    k = 0
    for i in input_vals:
        my_im_dict[k]= i
        k+=1

    print(my_im_dict)

    # preprocessing user input for the machine model
    for i in range(0,len(myInput)):
        if i < 2:
            myInput[i] = my_im_dict[i]

        elif i == 2:
            if my_im_dict[i] == "No": myInput[i] = 1
            else: myInput[i+1] = 1
        

        elif i == 3:
            if my_im_dict[i] == "No": myInput[i+1] = 1
            else: myInput[i+2] = 1
        
        

        elif i == 4:
            if my_im_dict[i] == "No": myInput[i+2] = 1
            else: myInput[i+3] = 1
        
        

        elif i == 5:
            if my_im_dict[i] == "seldom": myInput[i+3] = 1
            elif my_im_dict[i] == "often" : myInput[i+4] = 1
            else: myInput[i+5] = 1

    print(myInput)

    input_mat = [myInput]
    pred1 = model_name.predict(input_mat)
    return pred1[0]

dt, rft, knn = train_model(train_df)

result = single_prediction(rft)
print(result)




