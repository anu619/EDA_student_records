import numpy as np
import seaborn as sns
def dataSplit(df):
    # label encode final_grade
    from sklearn import preprocessing
    label_en = preprocessing.LabelEncoder()
    df.Final_grade = label_en.fit_transform(df.Final_grade)
    from sklearn.model_selection import train_test_split
    #Putting Feature Variable to X and Target variable to y.
    # Putting feature variable to X
    xdata = df.drop(['School', 'Sex', 'Age', 'Address', 'Family_size', 'Parents_status','Mother_job','Father_job','Reason',
                         'Guardian','Travel_time', 'School_support', 'Family_support', 'Paid_class', 'Activities','Nursery', 
                         'Higher_education', 'Internet', 'Romantic','Family_relation_quality', 'Free_time','Go_out','Weekday_alcohol_usage',
                         'Health', 'Absences', 'Final_grade'],axis=1)
    # Putting response variable to y
    ydata = df.Final_grade
    print(xdata)
    print(xdata.shape)
    print (ydata)
    print(ydata.shape)
    x_rus,y_rus = CIB(df)   
    
    xtrain_data,xtest_data,ytrain_data,ytest_data=train_test_split(x_rus,y_rus,test_size=0.25)
        
    return xtrain_data,xtest_data,ytrain_data,ytest_data, df

#Class Imbalance
def CIB(df):
    from collections import Counter
    from imblearn.over_sampling import SMOTE
    #from matplotlib import pyplot as fig
    import matplotlib.pyplot as fig
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter
    import collections

    X = df.drop('Final_grade',axis=1)
    y = df.Final_grade

    from imblearn.under_sampling import RandomUnderSampler
    from matplotlib import pyplot
    rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
    x_rus, y_rus = rus.fit_resample(X, y)
    print('Original dataset shape:', Counter(y))
    print('Resample dataset shape RUS', Counter(y_rus))
    
    cnt = collections.Counter(y)
    fig.title('Original dataset shape:')
    fig.bar(cnt.keys(), cnt.values())
    fig.show()
    
    cnt = collections.Counter(y_rus)
    fig.title('Resample dataset shape RUS')
    fig.bar(cnt.keys(), cnt.values())
    fig.show()
    return x_rus, y_rus 

def RFR(xtrain_data,xtest_data,ytrain_data,ytest_data):
    #Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
    from sklearn import metrics
    import matplotlib.pyplot as fig
    rfr_model = RandomForestRegressor(random_state=42)
    f_model = rfr_model.fit(xtrain_data, ytrain_data)
    ypred_data = f_model.predict(xtest_data)
    #Accuracy Measures: Evaluating Mean Squared Error, Mean Absolute Error and Root Mean Squared Error
    #  Calculating Mean Squared Error
    print('Classification Analysis Report')
    mean_sq_error = mean_squared_error(ytest_data, ypred_data)
    print('\nMean squared error :', round(mean_sq_error, 2))
     
    # Calculating Mean Absolute Error
    mean_abs_sq_error = mean_absolute_error(ytest_data, ypred_data)
    print('Mean absolute error :', round(mean_abs_sq_error, 2))
    # Calculating Root Mean Squared Error
    root_mean_sq_error = np.sqrt(mean_sq_error)
    print('Root Mean Squared Error :', round(root_mean_sq_error, 2))
    print('Accuracy Score of RFR: ',accuracy_score(ytest_data, ypred_data))
    print("Raondom Forest Regression Model Score" , ":" , f_model.score(xtrain_data, ytrain_data) , "," ,"Cross Validation Score" ,":" ,
          f_model.score(xtest_data, ytest_data))
    print('Confusion Matrix :',confusion_matrix(ytest_data,ypred_data))
    #Precision, recall for Multi-Class Classification
    matrix_data = classification_report(ytest_data,ypred_data,digits=3)
    print('Classification report : \n',matrix_data)
    
    return 0

def RFC(xtrain_data,xtest_data,ytrain_data,ytest_data):
    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
    from sklearn import metrics
    # Random forest classifier model
    model_rfc = RandomForestClassifier(n_estimators = 70,random_state=42)
    # fitting the model
    f_model=model_rfc.fit(xtrain_data, ytrain_data)
    ypred_data=model_rfc.predict(xtest_data)
    print('Classification Analysis Report')
    mean_sq_error = mean_squared_error(ytest_data, ypred_data)
    print('\nMean squared error :', round(mean_sq_error, 2))
    # Calculating Mean Absolute Error
    mean_abs_sq_error = mean_absolute_error(ytest_data, ypred_data)
    print('Mean absolute error :', round(mean_abs_sq_error, 2))
    # Calculating Root Mean Squared Error
    root_mean_sq_error = np.sqrt(mean_sq_error)
    print('Root Mean Squared Error :', round(root_mean_sq_error, 2))
    print('Accuracy Score of RFC:',metrics.accuracy_score(ytest_data,ypred_data))
    print("Raondom Forest Model Score" , ":" , f_model.score(xtrain_data, ytrain_data) , "," ,"Cross Validation Score" ,":" ,
          f_model.score(xtest_data,ytest_data))
    print(metrics.confusion_matrix(ytest_data,ypred_data))
    #Precision, recall for Multi-Class Classification
    matrix_data = metrics.classification_report(ytest_data,ypred_data,digits=3)
    print('Classification report : \n',matrix_data)
    cf_matrix_rfc = metrics.confusion_matrix(ytest_data,ypred_data)
    hmap_rfc=sns.heatmap(cf_matrix_rfc, annot=True)
    hmap_rfc.set(title="Random forest classifier Model")
    
    return 0

def SuppotVM(xtrain_data,xtest_data,ytrain_data,ytest_data):
    #classification using SVM
    from sklearn.svm import SVC
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
    from sklearn import metrics
    # Instantiate the Support Vector Classifier (SVC)
    model_svc = SVC(C=1.0, random_state=1, kernel='linear')
    # Fit the model
    f_model=model_svc.fit(xtrain_data, ytrain_data)
    # Make the predictions
    ypred_data = model_svc.predict(xtest_data)
    # Measure the performance
    print('Classification Analysis Report')
    mean_sq_error = mean_squared_error(ytest_data, ypred_data)
    print('\nMean squared error :', round(mean_sq_error, 2))
    # Calculating Mean Absolute Error
    mean_abs_sq_error = mean_absolute_error(ytest_data, ypred_data)
    print('Mean absolute error :', round(mean_abs_sq_error, 2))
    # Calculating Root Mean Squared Error
    root_mean_sq_error = np.sqrt(mean_sq_error)
    print('Root Mean Squared Error :', round(root_mean_sq_error, 2))
    print('Accuracy Score of SVM:',metrics.accuracy_score(ytest_data,ypred_data))
    print("SVM Model Score" , ":" , f_model.score(xtrain_data, ytrain_data) , "," ,"Cross Validation Score" ,":" ,
          f_model.score(xtest_data,ytest_data))
    print(metrics.confusion_matrix(ytest_data,ypred_data))
    #Precision, recall for Multi-Class Classification
    matrix_data = metrics.classification_report(ytest_data,ypred_data,digits=3)
    print('Classification report : \n',matrix_data)
    cf_matrix_svm = metrics.confusion_matrix(ytest_data,ypred_data)
    hmap_svm=sns.heatmap(cf_matrix_svm, annot=True)
    hmap_svm.set(title="Support vector machine Model")
    return 0

def MLPC(xtrain_data,xtest_data,ytrain_data,ytest_data):
    #classification using MlPC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
    from sklearn import metrics
    #Initializing the MLPClassifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    f_model=mlp_classifier.fit(xtrain_data, ytrain_data)
    ypred_data = mlp_classifier.predict(xtest_data)
    # Measure the performance
    print('Classification Analysis Report')
    mean_sq_error = mean_squared_error(ytest_data, ypred_data)
    print('\nMean squared error :', round(mean_sq_error, 2))
    # Calculating Mean Absolute Error
    mean_abs_sq_error = mean_absolute_error(ytest_data, ypred_data)
    print('Mean absolute error :', round(mean_abs_sq_error, 2))
    # Calculating Root Mean Squared Error
    root_mean_sq_error = np.sqrt(mean_sq_error)
    print('Root Mean Squared Error :', round(root_mean_sq_error, 2))
    print('Accuracy Score of MLP:',metrics.accuracy_score(ytest_data,ypred_data))
    print("MLP Model Score" , ":" , f_model.score(xtrain_data, ytrain_data) , "," ,"Cross Validation Score" ,":" ,
          f_model.score(xtest_data,ytest_data))
    print(metrics.confusion_matrix(ytest_data,ypred_data))
    #Precision, recall for Multi-Class Classification
    matrix_data = metrics.classification_report(ytest_data,ypred_data,digits=3)
    print('Classification report : \n',matrix_data)
    cf_matrix_mplc = metrics.confusion_matrix(ytest_data,ypred_data)
    hmap_mlpc=sns.heatmap(cf_matrix_mplc, annot=True)
    hmap_mlpc.set(title="Multi-Layer Perceptron Neural Networks Model")
   