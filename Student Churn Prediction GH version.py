# -*- coding: utf-8 -*-
"""
Alternative offer withdrawal prediction model

- Trained on withdrawal data of students given places OTHER than the one they wanted
- Model utilises stacked generalisation to create a meta model (currently logistic regression)
and hyperparameter tuning (gridsearch for full exploration, though Bayesian recommended for speed)

NOTE: GDPR COMPLIANT AND ANONYMISED FOR GITHUB
"""

# =============================================================================
# #Loading and shaping data
# =============================================================================

#load dependencies
import os
import pandas as pd
import numpy as np

#set up directory
os.getcwd()
os.chdir('')
os.listdir(os.getcwd())

#load admissions data, discard extraneous columns, and rename ambiguous columns
admissions_data=pd.read_csv('Admissions Data.csv')
admissions_data=admissions_data[['Student Code','pre_nqa.DRL_IDRC','pre_nqa.drl_timestamp','min_nqa.DRL_IDRC','min_nqa.drl_timestamp','pre_nqa.DRL_MCRC','min_nqa.DRL_MCRC']]
admissions_data.rename(columns={'pre_nqa.DRL_IDRC':'application_admissions_code','min_nqa.DRL_IDRC':'acceptance_admissions_code','pre_nqa.drl_timestamp':'application_timestamp','min_nqa.drl_timestamp':'acceptance_timestamp','pre_nqa.DRL_MCRC':'application_discipline_code','min_nqa.DRL_MCRC':'acceptance_discipline_code'},inplace=True)

#function to add "_timestamp" to a value from a list, convert the column with that string name to datetime, then augment that column
def columns_to_datetime(df,stamps,augments):
    for i in stamps:
        column=str(i)+'_timestamp'
        df[column]=pd.to_datetime(df[column])    
        for j in augments:
            augmented_column_name=str(i)+'_'+str(j)
            if j == 'weekday':
                df[augmented_column_name]=(df[column].dt.dayofweek)+1
            if j == 'month':
                df[augmented_column_name]=df[column].dt.month
            if j == 'year':
                df[augmented_column_name]=df[column].dt.year
            if j == 'hour':
                df[augmented_column_name]=df[column].dt.hour  

#define columns and augments
timestamps_to_augment=['application','acceptance']
augments=['weekday','month','year','hour']

#deploy function
columns_to_datetime(admissions_data,timestamps_to_augment,augments)
del(timestamps_to_augment,augments)

#augment admissions data by :
    # extracting if application/offer was conditional/unconditional
    # if it was the students first choice or their insurance choice
    # indicating if the discipline codes differ between application and acceptance
    # if application campus differs from acceptance campus
admissions_data['firstchoice_application'] = np.where(admissions_data['application_admissions_code'].str[1:2] == 'F', 1, 0)
admissions_data['unconditional_application'] = np.where(admissions_data['application_admissions_code'].str[:1] == 'U', 1, 0)
admissions_data['discipline_code_disparity']=np.where(admissions_data['application_discipline_code'] == admissions_data['acceptance_discipline_code'], 1, 0)

#grab alternative offer flag and campus applied for/accepted from student dataset
student_data=pd.read_csv('Student Data.csv')
student_data=student_data[student_data['Alternative Offer Marker'] == 'Y']
student_data.rename(columns={'pre_nqa.Campus':'Campus_Chosen','Campus Group':'Campus_Given'},inplace=True)
combined_df=pd.merge(student_data,admissions_data,how='left',on='Student Code')
del(admissions_data,student_data)

#highlight if the campus differs from what the student applied for
combined_df['Alternative or original campus'] = np.where(combined_df['Alternative or original campus'] == 'Studying at campus applied to ', 1, 0)

#save
combined_df.to_csv('combined_df.csv',index=False)

# =============================================================================
# # Diagnostic analysis (why it happened) 
# =============================================================================

#file operating
import os
import pandas as pd
import pickle

#for data shaping
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import OrdinalEncoder

#training models
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner
#from sklearn.metrics import brier_score_loss

#for visualisation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report

#ordinally encode features and resample dataset if unbalanced
def encode_and_resample(X,y):
    for column in X:
        if (X[column].isna().sum() >0):
            print(column+" has: "+str(X[column].isna().sum()) + " NAs")
    X_col_names=X.columns
    oe = OrdinalEncoder()
    oe.fit(X)
    X = oe.transform(X)
    X=pd.DataFrame(data=X,columns=X_col_names)
    prior=y.value_counts()
    if prior.iloc[0]/prior.iloc[1]>3: #can't find a definition of imbalanced, so going for 1:3
        print("dataset imbalanced")
    sampler = SMOTEN(random_state=0) #SmoteNC works on categorical and numerical (different algo!), but i've already encoded - I might be missing something here
    X, y = sampler.fit_resample(X, y)
    posterior=y.value_counts()
    if posterior.iloc[0]==posterior.iloc[1]:
        print("dataset now balanced")
    X=X.to_numpy() #out of fold predictions function needs arrary
    y=y.to_numpy()
    return(X,y)

#extract features and target, then encode and resample if required, then split into training and testing
def prep_features_and_targets(df,cols_to_remove,target):
    X=df
    for i in cols_to_remove:
        X=X.drop([i],axis=1)
    X_cols=X.columns
    y = df[target].apply(lambda x: 1 if x=='Y' else 0)
    X,y=encode_and_resample(X,y) 
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
    return(X, X_val, y, y_val,X_cols)

# create a list of base-models     
#note only a few fields are utilised, but we have 100s of tables so deep learning could be valuable!
def get_models():
    models = list()
    models.append(LogisticRegression(solver='liblinear'))
    models.append(DecisionTreeClassifier())
    models.append(SVC(gamma='scale', probability=True))
    models.append(GaussianNB())
    models.append(KNeighborsClassifier())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier(n_estimators=10))
    models.append(RandomForestClassifier(n_estimators=10))
    models.append(ExtraTreesClassifier(n_estimators=10))
    return models

# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))
    # add base models
    models = get_models()
    ensemble.add(models)
    
    #fine best hyperparameters for meta_model
    search_space ={'solver': ['newton-cg', 'lbfgs','sag', 'saga'],
    'penalty': ['l2', 'none'],       #not all penalties work with all solvers - I have lengthier code to implement more combinations in progress
    'C' : [0.01, 0.1, 1],
    'max_iter': [50, 100, 200]}
    tuned_model = GridSearchCV(estimator=LogisticRegression(),
                                param_grid=search_space,
                                cv=5,
                                scoring='accuracy')
    tuned_model.fit(X, y) #ignore this error
    params=tuned_model.best_params_
    tuned_model=LogisticRegression(C=params.get('C'),
    solver=params.get('solver'),
    penalty=params.get('penalty'), 
    max_iter=params.get('max_iter'))

    # add the meta model
    ensemble.add_meta(tuned_model)
    return ensemble

# create the super learner
def get_super_learner_base(X):
    ensemble_base = SuperLearner(scorer=accuracy_score, folds=10, shuffle=True, sample_size=len(X))
    # add base models
    models = get_models()
    ensemble_base.add(models)

    # add the meta model
    ensemble_base.add_meta(LogisticRegression(solver='lbfgs'))
    return ensemble_base

#plot confusion matrix as well as classification report #NOTE NEEDS SOME FORMATTING WORK
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    
    #table
    plt.subplot(224)
    report=classification_report(y_val, yhat,target_names=['didnt withdraw','withdrew'],output_dict=True) #ignore this error
    df = pd.DataFrame(report).transpose()
    cell_text = []
    for row in range(len(df)):
        cell_text.append(df.iloc[row])
    plt.table(cellText=cell_text,rowLabels=df.index, colLabels=df.columns, loc='center')
    plt.axis('off')
    
    #cm
    plt.subplot(222)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
      horizontalalignment="center",
      color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
#set up directory
os.getcwd()
os.chdir('')
combined_df=pd.read_csv('combined_df.csv')
   
#prepare data     
cols_to_remove=['application_year','acceptance_year','Count','Alternative Offer Marker',
                                     'Withdrawal Reason Group',
                                     'Programme Code', 
                                     #'Current Sub-Discipline',#putting on seperate lines to make it easier to comment out during model tuning
                                     'application_discipline_code',
                                     'acceptance_discipline_code',
                                     'Withdrawal Marker', 'Student Code', 
                                     'Cohort Year', 'application_admissions_code','application_timestamp', 
                                     'acceptance_admissions_code','acceptance_timestamp']

X, X_val, y, y_val,X_cols=prep_features_and_targets(combined_df,cols_to_remove,'Withdrawal Marker')
del(cols_to_remove)

#create super learner and predict data
ensemble= get_super_learner(X)
ensemble.fit(X, y)
data=pd.DataFrame(ensemble.data) #summarize base learners
yhat = ensemble.predict(X_val)

#assess loss (Brier score) and accuracy of tuned/best base model
# print('Tuned Super Learner Brier Score: %.3f' % (brier_score_loss(y_val, yhat)))
# print('Super Learner Accuracy: %.3f' % (accuracy_score(y_val, yhat) * 100))
# print(str(data.sort_values('score-m', ascending=False).index[0])+': %.3f' % (data.sort_values('score-m', ascending=False).iat[0,0]*100))

#clean up
del(X_cols,y,X_val,data,X)

#visualise confusion matrix
cm = confusion_matrix(y_true=y_val, y_pred=yhat) 
cm_plot_labels = ['Didnt withdraw', 'Withdrew']
#fig.savefig('./confusion matrix.png', bbox_inches='tight')
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
del(cm,cm_plot_labels,yhat,y_val)

#save model
filename = 'finalized_model.sav' #could include dynamic date entry so that we always have older versions
pickle.dump(ensemble, open(filename, 'wb'))
del(ensemble,filename,combined_df)

# =============================================================================
# # Predictive analysis (what will happen) 
# =============================================================================

import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from mlens.ensemble import SuperLearner

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
del(filename)

#set up directory
os.getcwd()
os.chdir('')
os.listdir(os.getcwd())

#load new data
admissions_data=pd.read_csv('newdata.csv')
admissions_data=admissions_data[['Student Code','pre_nqa.DRL_IDRC','pre_nqa.drl_timestamp','min_nqa.DRL_IDRC','min_nqa.drl_timestamp','pre_nqa.DRL_MCRC','min_nqa.DRL_MCRC']]
admissions_data.rename(columns={'pre_nqa.DRL_IDRC':'application_admissions_code','min_nqa.DRL_IDRC':'acceptance_admissions_code','pre_nqa.drl_timestamp':'application_timestamp','min_nqa.drl_timestamp':'acceptance_timestamp','pre_nqa.DRL_MCRC':'application_discipline_code','min_nqa.DRL_MCRC':'acceptance_discipline_code'},inplace=True)

#function to add "_timestamp" to a value from a list, convert the column with that string name to datetime, then augment that column
def columns_to_datetime(df,stamps,augments):
    for i in stamps:
        column=str(i)+'_timestamp'
        df[column]=pd.to_datetime(df[column])    
        for j in augments:
            augmented_column_name=str(i)+'_'+str(j)
            if j == 'weekday':
                df[augmented_column_name]=(df[column].dt.dayofweek)+1
            if j == 'month':
                df[augmented_column_name]=df[column].dt.month
            if j == 'year':
                df[augmented_column_name]=df[column].dt.year
            if j == 'hour':
                df[augmented_column_name]=df[column].dt.hour  

#define columns and augments
timestamps_to_augment=['application','acceptance']
augments=['weekday','month','year','hour']

#deploy function
columns_to_datetime(admissions_data,timestamps_to_augment,augments)
del(timestamps_to_augment,augments)

#augment admissions data by :
    # extracting if application/offer was conditional/unconditional
    # if it was the students first choice or their insurance choice
    # indicating if the discipline codes differ between application and acceptance
    # if application campus differs from acceptance campus
admissions_data['firstchoice_application'] = np.where(admissions_data['application_admissions_code'].str[1:2] == 'F', 1, 0)
admissions_data['unconditional_application'] = np.where(admissions_data['application_admissions_code'].str[:1] == 'U', 1, 0)
admissions_data['discipline_code_disparity']=np.where(admissions_data['application_discipline_code'] == admissions_data['acceptance_discipline_code'], 1, 0)

#remove unused columns
cols_to_remove=['application_year','acceptance_year','Alternative Offer Marker',
                                     'Programme Code', 
                                     #'Current Sub-Discipline',#putting on seperate lines to make it easier to comment out during model tuning
                                     'application_discipline_code',
                                     'acceptance_discipline_code',
                                     'Student Code', 
                                     'Cohort Year', 'application_admissions_code','application_timestamp', 
                                     'acceptance_admissions_code','acceptance_timestamp']

#extract features then encode
def prep_features_for_testing(df,cols_to_remove):
    X=df
    for i in cols_to_remove:
        X=X.drop([i],axis=1)
    X_col_names=X.columns
    oe = OrdinalEncoder()
    oe.fit(X)
    X = oe.transform(X)
    X=pd.DataFrame(data=X,columns=X_col_names)
    return(X)
X=prep_features_for_testing(admissions_data,cols_to_remove)
del(cols_to_remove)

#Predict withdrawals and save results
admissions_data['Withdrawal_Marker']=loaded_model.predict(X)
admissions_data.to_csv('./predicted_data/results_df.csv',index=False)
del(X,admissions_data,loaded_model)

# =============================================================================
# # Prescriptive analysis (what steps to take)
# =============================================================================

import os
import pandas as pd
import numpy as np
import yagmail

#set up directory
os.getcwd()
os.chdir('')
os.listdir(os.getcwd())
predicted_df=pd.read_csv('./predicted_data/results_df.csv')

#generate a list of IDs who are predicted to withdraw and send email containing these
predicted_withdrawal_id=predicted_df[predicted_df['Withdrawal_Marker']==1]['Student Code']
predicted_withdrawal_id.to_csv('IDs of students predicted to withdraw.csv',index=False)
del(predicted_withdrawal_id)

#generate graphs for withdrawals
predicted_df_all=predicted_df
predicted_df=predicted_df[predicted_df['Withdrawal_Marker']==1]

#remove for github anonymising
predicted_df['Current College'] = (predicted_df.groupby(['Current College']).ngroup()).astype(str)
predicted_df['Current Sub-Discipline'] = (predicted_df.groupby(['Current Sub-Discipline']).ngroup()).astype(str)
predicted_df['Campus Group'] = (predicted_df.groupby(['Campus Group']).ngroup()).astype(str)

#loop to create graphs to email out
for i in ['Current College','Current Sub-Discipline','Campus Group']:
    variable=i
    coll_withdrawals=(predicted_df.groupby(['Withdrawal Marker',variable]).size().reset_index(name='counts')).sort_values(variable, ascending=True)
    coll_withdrawals=coll_withdrawals.pivot(variable, 'Withdrawal Marker', "counts").sort_values(1, ascending=False)
    coll_withdrawals = coll_withdrawals.rename_axis(variable).reset_index()
    
    withdrawals = coll_withdrawals[1]
    colors = ['orange' if (h < max(withdrawals)) else 'red' for h in withdrawals]
    
    y1 = coll_withdrawals[1]
    x = np.arange(len(coll_withdrawals))
    
    width=0.5
    fig, ax = plt.subplots(figsize=(10,5))
    for i,j in zip(x,y1):
        ax.annotate(str(j),xy=(i,j),ha='center', va='bottom',fontsize=9)
    plt.bar(coll_withdrawals[variable],coll_withdrawals[1],width=width,label='Predicted withdraw',color=colors) 
    plt.xlabel(variable)
    plt.ylabel('Predicted withdrawals')
    plt.title('How Many Alternative Offer Students Withdrew per '+variable+'?')
    fig.savefig('./'+variable+'_predicted_withdrawals.png', bbox_inches='tight')
    plt.show()    
del(x,y1,colors,withdrawals,coll_withdrawals,fig,ax,i,j,width)

introtext=('Out of '+str(pd.DataFrame(predicted_df['Withdrawal Marker'].value_counts()).iat[0,0])+' entries, '+str(pd.DataFrame(predicted_df['Withdrawal Marker'].value_counts()).iat[1,0])+' are predicted to be at risk of withdrawing')

messagedata=pd.DataFrame(columns=[''])
for i in ['Current College','Current Sub-Discipline','Campus Group']:
    variable=i
    coll_withdrawals=(predicted_df.groupby(['Withdrawal Marker',variable]).size().reset_index(name='counts')).sort_values(variable, ascending=True)
    coll_withdrawals=coll_withdrawals.pivot(variable, 'Withdrawal Marker', "counts").sort_values(1, ascending=False)
    coll_withdrawals = coll_withdrawals.rename_axis(variable).reset_index()
    messagedata.loc[len(messagedata)] =[('')] 
    messagedata.loc[len(messagedata)] =[('At a '+ str(variable)+ ' breakdown:')]   
    for i in range(len(coll_withdrawals)):
        messagedata.loc[len(messagedata)] =[(str(variable)+' '+str(coll_withdrawals.iat[i,0])+': '+str(coll_withdrawals.iat[i,1]))]
messagedata.drop(index=messagedata.index[0], axis=0, inplace=True)
messagedata=messagedata.to_string(index=False)

yag = yagmail.SMTP("", oauth2_file="")

to = 'n'
subject = 'Withdrawal Prediction'
img1 = './Campus Group_predicted_withdrawals.png'
img2='./Current Sub-Discipline_predicted_withdrawals.png'
img3='./Current College_predicted_withdrawals.png'

#################################### ANONYMISED FOR GITHUB #################################
body=f"""
Dear Monitor,

Attached are student codes that have been flagged as being predicted to withdraw. Please contact Student Records for identification details

{introtext}
{messagedata}

Please see the attached graphs for more information    
    
Regards,
Nathan
"""
yag.send(to = to, subject = subject, contents = [body,img1,img2,img3])
