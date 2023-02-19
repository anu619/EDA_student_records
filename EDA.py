import pandas as pd

def dataLoad():
    #loading the dataset student-mat.csv
    try:
        file1='student-mat.csv'
        dataset1=pd.read_csv('student-mat.csv',sep=';')
        #loading the dataset student-mat.csv
        file2='student-por.csv'
        dataset2=pd.read_csv('student-por.csv',sep=';')
        #Merging the two CSV files
        data=pd.concat([dataset1,dataset2])
        #Renaming Column labels  
        data.columns=['School', 'Sex', 'Age', 'Address', 'Family_size', 'Parents_status', 'Mother_education', 'Father_education',
               'Mother_job', 'Father_job', 'Reason', 'Guardian', 'Travel_time', 'Study_time',
               'Failures', 'School_support', 'Family_support', 'Paid_class', 'Activities', 'Nursery',
               'Higher_education', 'Internet', 'Romantic', 'Family_relation_quality', 'Free_time', 'Go_out', 'Workday_alcohol_usage',
               'Weekday_alcohol_usage', 'Health', 'Absences', 'Period1_score', 'Period2_score', 'Final_score']
        #print(data.columns)
        # Creating an additional attribute Final_Grade based on Final_Score 
        # Attribute Values: High:15-20 Medium:10-14 Low:0-9
        # High => 2, Medium => 1, Low => 0
        data['Final_grade'] = 0
        data.loc[(data.Final_score >= 15) & (data.Final_score <= 20), 'Final_grade']=2 
        data.loc[(data.Final_score >= 10) & (data.Final_score <= 14), 'Final_grade']=1 
        data.loc[(data.Final_score >= 0) & (data.Final_score <= 9), 'Final_grade']=0
        return data
    except IOError:
            print('file not found')
        
def checkNotnull(df):
    #Exploratory Data Analysis
    #Shows the missing value information
    print(df.isnull().any())   
    

def statInfo(df):
    #Statistical analysis of the dataset
    mean1=df.mean(numeric_only=True)
    median1=df.median(numeric_only=True)
    stddev=df.std(numeric_only=True)
    max1=df.max(numeric_only=True)
    min1=df.min(numeric_only=True)
    var1=df.var(numeric_only=True)
    skew1=df.skew(numeric_only=True)
    kurto1=df.kurtosis(numeric_only=True)
    statdata = [mean1,median1,stddev,max1,min1,var1,skew1,kurto1]
    stat_desc=pd.concat(statdata,axis=1, join='inner')
    stat_desc.columns = ['Mean', 'Median', 'Standard Deviation', 'Maximum','Minimum', 'Variance', 'Skewness','Kurtosis']
    print(stat_desc)    

def mostCorr(df):
    # Find correlation of attributes with the Final Score
    mcr = df.corr().abs()['Final_score'].sort_values(ascending=False)
    # Top 8 most correlation features with Grade
    mcr = mcr[:9]
    print(mcr)  
    


