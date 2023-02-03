# -*- coding: utf-8 -*-
"""
Student Churn Model

Trained on withdrawal data of students given places OTHER than the one they wanted
Train model is then applied to current student data to predict who will withdraw and where further attention is required

"""

# =============================================================================
# #Loading and shaping data
# =============================================================================

#load dependencies
import os
import pandas as pd
import numpy as np
#for EDA
import matplotlib.pyplot as plt
#for model training
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
#model fitting
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report
# distributing predictive analytics
import yagmail

#set up directory
os.getcwd()
os.chdir('')
os.listdir(os.getcwd())

#load admissions data and discard extraneous columns
admissions_data=pd.read_csv('./training_data/Admissions Data.csv')
admissions_data=admissions_data[['Student Code','pre_nqa.DRL_IDRC','pre_nqa.drl_timestamp','min_nqa.DRL_IDRC','min_nqa.drl_timestamp','pre_nqa.DRL_MCRC','min_nqa.DRL_MCRC']]
admissions_data.rename(columns={'pre_nqa.DRL_IDRC':'application_admissions_code','min_nqa.DRL_IDRC':'acceptance_admissions_code','pre_nqa.drl_timestamp':'application_timestamp','min_nqa.drl_timestamp':'acceptance_timestamp','pre_nqa.DRL_MCRC':'application_discipline_code','min_nqa.DRL_MCRC':'acceptance_discipline_code'},inplace=True)

#augment admissions data with day(in week)/month/hour of application/acceptance
admissions_data['application_timestamp'] = pd.to_datetime(admissions_data['application_timestamp'])
admissions_data['acceptance_timestamp'] = pd.to_datetime(admissions_data['acceptance_timestamp'])
admissions_data['application_weekday'] = (admissions_data['application_timestamp'].dt.dayofweek)+1
admissions_data['application_month'] = admissions_data['application_timestamp'].dt.month
admissions_data['application_year'] = admissions_data['application_timestamp'].dt.year
admissions_data['application_hour'] = admissions_data['application_timestamp'].dt.hour
admissions_data['acceptance_weekday'] = (admissions_data['acceptance_timestamp'].dt.dayofweek)+1
admissions_data['acceptance_month'] = admissions_data['acceptance_timestamp'].dt.month
admissions_data['acceptance_year'] = admissions_data['acceptance_timestamp'].dt.year
admissions_data['acceptance_hour'] = admissions_data['acceptance_timestamp'].dt.hour

#augment admissions data by :
    # extracting if application/offer was conditional/unconditional
    # if it was the students first choice or their insurance choice
    # indicating if the discipline codes differ between application and acceptance
    # if application campus differs from acceptance campus
admissions_data['firstchoice_application'] = np.where(admissions_data['application_admissions_code'].str[1:2] == 'F', 1, 0)
admissions_data['unconditional_application'] = np.where(admissions_data['application_admissions_code'].str[:1] == 'U', 1, 0)
admissions_data['discipline_code_disparity']=np.where(admissions_data['application_discipline_code'] == admissions_data['acceptance_discipline_code'], 1, 0)

#load student data, filter to relevant students, and merge admissions data
student_data=pd.read_csv('./training_data/Student Data.csv')
student_data=student_data[student_data['Alternative Offer Marker'] == 'Y'] #we're only looking at these
student_data.rename(columns={'pre_nqa.Campus':'Campus_Chosen','Campus Group':'Campus_Given'},inplace=True) #making column names clearer
combined_df=pd.merge(student_data,admissions_data,how='left',on='Student Code')
combined_df['Alternative or original campus'] = np.where(combined_df['Alternative or original campus'] == 'Studying at campus applied to ', 1, 0) #turning lengthy strings into a 1 or 0
del(admissions_data,student_data)

# =============================================================================
# Descriptive Analysis (What happened?)
# =============================================================================

#quick check to see how many withdrawals we have
combined_df['Withdrawal Marker'].value_counts()
quickstat=combined_df.groupby('Withdrawal Marker').mean()
del(quickstat)

#total withdrawals per year chart plot 
acyr_withdrawals=(combined_df.groupby(['Withdrawal Marker','Cohort Year']).size().reset_index(name='counts')).sort_values('Cohort Year', ascending=True)
acyr_withdrawals=acyr_withdrawals.pivot('Cohort Year', 'Withdrawal Marker', "counts")
acyr_withdrawals = acyr_withdrawals.rename_axis('Cohort Year').reset_index()

withdrawals = acyr_withdrawals['Y']
colors = ['orange' if (h < max(withdrawals)) else 'red' for h in withdrawals]

y1 = acyr_withdrawals['Y']
x = np.arange(len(acyr_withdrawals))
width=0.5
fig, ax = plt.subplots(figsize=(10,5))
for i,j in zip(x,y1):
    ax.annotate(str(j),xy=(i,j),ha='center', va='bottom',fontsize=9)
plt.bar(acyr_withdrawals['Cohort Year'],acyr_withdrawals['N'],width=width,label='Remained') 
plt.bar(acyr_withdrawals['Cohort Year'],acyr_withdrawals['Y'],width=width,label='Withdrew',color=colors) 
plt.legend(frameon=True, fontsize=10)
plt.xlabel('Cohort Year')
plt.ylabel('Withdrawals')
plt.title('How Many Alternative Offer Students Withdrew per Cohort Year?')
fig.savefig('./training_graphs/acyr_withdrawals.png', bbox_inches='tight')
plt.show()

#withdrawals per college chart plot
coll_withdrawals=(combined_df.groupby(['Withdrawal Marker','Current College']).size().reset_index(name='counts')).sort_values('Current College', ascending=True)
coll_withdrawals=coll_withdrawals.pivot('Current College', 'Withdrawal Marker', "counts")
coll_withdrawals = coll_withdrawals.rename_axis('College').reset_index()

withdrawals = coll_withdrawals['Y']
colors = ['orange' if (h < max(withdrawals)) else 'red' for h in withdrawals]

y1 = coll_withdrawals['Y']
x = np.arange(len(coll_withdrawals))

width=0.5
fig, ax = plt.subplots(figsize=(10,5))
for i,j in zip(x,y1):
    ax.annotate(str(j),xy=(i,j),ha='center', va='bottom',fontsize=9)
plt.bar(coll_withdrawals['College'],coll_withdrawals['N'],width=width,label='Remained') 
plt.bar(coll_withdrawals['College'],coll_withdrawals['Y'],width=width,label='Withdrew',color=colors) 
plt.legend(frameon=True, fontsize=10)
plt.xlabel('College')
plt.ylabel('Withdrawals')
plt.title('How Many Alternative Offer Students Withdrew per College?')
fig.savefig('./training_graphs/coll_withdrawals.png', bbox_inches='tight')
plt.show()

#withdrawals per campus (from each campus) chart plot
camp_withdrawals=combined_df[combined_df['Withdrawal Marker']=='Y']
camp_withdrawals=(camp_withdrawals.groupby(['Campus_Given','Campus_Chosen']).size().reset_index(name='counts')).sort_values('Campus_Chosen', ascending=True)
camp_withdrawals=camp_withdrawals.pivot('Campus_Given', 'Campus_Chosen', "counts")
camp_withdrawals = camp_withdrawals.rename_axis('Campus_Chosen').reset_index()

y1 = camp_withdrawals[1]
x = np.arange(len(camp_withdrawals))

width=0.5
fig, ax = plt.subplots(figsize=(10,5))
for i,j in zip(x,y1):
    ax.annotate(str(j),xy=(i,j),ha='center', va='bottom',fontsize=9)
plt.bar(camp_withdrawals['Campus_Chosen'],camp_withdrawals[1],width=width,label='1 Given',color=colors) 
plt.legend(frameon=True, fontsize=10)
plt.xlabel('Campus Chosen')
plt.ylabel('Withdrawals')
plt.title('How Did Withdrawals Differ Between Chosen and Given College??')
fig.savefig('./training_graphs/camp_withdrawals.png', bbox_inches='tight')
plt.show()

#what time of day do students submit and accept their application chart plot
time_withdrawals=combined_df[['Student Code','acceptance_hour']]
time_withdrawals['stage']='acceptance'
time_withdrawals.rename(columns={'acceptance_hour':'hour'},inplace=True)
time_withdrawals_app=combined_df[['Student Code','application_hour']]
time_withdrawals_app['stage']='application'
time_withdrawals_app.rename(columns={'application_hour':'hour'},inplace=True)
time_withdrawals=time_withdrawals.append(time_withdrawals_app)
del(time_withdrawals_app)

time_withdrawals=(time_withdrawals.groupby(['hour','stage']).size().reset_index(name='counts')).sort_values('hour', ascending=True)
time_withdrawals=time_withdrawals.pivot('hour', 'stage', "counts").plot(kind='bar')
plt.show()

#dayofweek chart plot
day_withdrawals=combined_df[['Student Code','acceptance_weekday']]
day_withdrawals['stage']='acceptance'
day_withdrawals.rename(columns={'acceptance_weekday':'weekday'},inplace=True)
day_withdrawals_app=combined_df[['Student Code','application_weekday']]
day_withdrawals_app['stage']='application'
day_withdrawals_app.rename(columns={'application_weekday':'weekday'},inplace=True)
day_withdrawals=day_withdrawals.append(day_withdrawals_app)
del(day_withdrawals_app)
day_withdrawals=(day_withdrawals.groupby(['weekday','stage']).size().reset_index(name='counts')).sort_values('weekday', ascending=True)
day_withdrawals=day_withdrawals.pivot('weekday', 'stage', "counts").plot(kind='bar')
plt.show()

#month chart plot
month_withdrawals=combined_df[['Student Code','acceptance_month']]
month_withdrawals['stage']='acceptance'
month_withdrawals.rename(columns={'acceptance_month':'month'},inplace=True)
month_withdrawals_app=combined_df[['Student Code','application_month']]
month_withdrawals_app['stage']='application'
month_withdrawals_app.rename(columns={'application_month':'month'},inplace=True)
month_withdrawals=month_withdrawals.append(month_withdrawals_app)
del(month_withdrawals_app)

month_withdrawals=(month_withdrawals.groupby(['month','stage']).size().reset_index(name='counts')).sort_values('month', ascending=True)
month_withdrawals=month_withdrawals.pivot('month', 'stage', "counts").plot(kind='bar')
plt.show()

#clean up
del(acyr_withdrawals,camp_withdrawals,coll_withdrawals,day_withdrawals,time_withdrawals,month_withdrawals)

# =============================================================================
# # Diagnostic analysis (why it happened) 
# =============================================================================

#split df into (dummy) features and target
X = pd.get_dummies(combined_df.drop(['application_year','acceptance_year','Count','Alternative Offer Marker', #all students are alternative offer holders
                                     'Withdrawal Reason Group', 
                                     'Programme Code', 
                                     #'Current Sub-Discipline',#putting on seperate lines to make it easier to comment out during model tuning
                                     'application_discipline_code',
                                     'acceptance_discipline_code',
                                     'Withdrawal Marker', 'Student Code', 
                                     'Cohort Year', 'application_admissions_code','application_timestamp', 
                                     'acceptance_admissions_code','acceptance_timestamp'], axis=1))
y = combined_df['Withdrawal Marker'].apply(lambda x: 1 if x=='Y' else 0)

#SMOTE minor class balancer #https://arxiv.org/pdf/1106.1813.pdf due to imbalanced classes
os = SMOTE(random_state=36)
X,y = os.fit_resample(X, y)
y = pd.DataFrame(data=y,columns=['Withdrawal Marker'])
del(os)

#split df into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=36,stratify=y)

#define model and compile
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

#fit
model.fit(X_train, y_train, epochs=200, batch_size=32)

#predict
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
accuracy_score(y_test, y_hat)

#add confusion matrix from sci-kit learn's site:
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
 plt.imshow(cm, interpolation='nearest', cmap=cmap)
 plt.title(title)
 plt.colorbar()
 tick_marks = np.arange(len(classes))
 plt.xticks(tick_marks, classes, rotation=45)
 plt.yticks(tick_marks, classes)
 
cm = confusion_matrix(y_true=y_test, y_pred=y_hat) 
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
 plt.text(j, i, cm[i, j],
 horizontalalignment="center",
 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
cm_plot_labels = ['Didnt withdraw', 'Withdrew']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.show()
del(cm_plot_labels,i,j,thresh)

#print report
print(classification_report(y_test, y_hat))

#save model if it's performing well
model.save('ddmmyyyy_alternative_offer_withdrawal_tf_prediction')

del(X,X_test,X_train,y,y_hat,y_test,y_train)

# =============================================================================
# # Predictive analysis (what will happen?) 
# =============================================================================

#load new years data and discard extraneous columns
admissions_data=pd.read_csv('New Admisions Data.csv')
admissions_data=admissions_data[['Student Code','pre_nqa.DRL_IDRC','pre_nqa.drl_timestamp','min_nqa.DRL_IDRC','min_nqa.drl_timestamp','pre_nqa.DRL_MCRC','min_nqa.DRL_MCRC']]
admissions_data.rename(columns={'pre_nqa.DRL_IDRC':'application_admissions_code','min_nqa.DRL_IDRC':'acceptance_admissions_code','pre_nqa.drl_timestamp':'application_timestamp','min_nqa.drl_timestamp':'acceptance_timestamp','pre_nqa.DRL_MCRC':'application_discipline_code','min_nqa.DRL_MCRC':'acceptance_discipline_code'},inplace=True)

#augment admissions data with day(in week)/month/hour of application/acceptance
admissions_data['application_timestamp'] = pd.to_datetime(admissions_data['application_timestamp'])
admissions_data['acceptance_timestamp'] = pd.to_datetime(admissions_data['acceptance_timestamp'])
admissions_data['application_weekday'] = (admissions_data['application_timestamp'].dt.dayofweek)+1
admissions_data['application_month'] = admissions_data['application_timestamp'].dt.month
admissions_data['application_year'] = admissions_data['application_timestamp'].dt.year
admissions_data['application_hour'] = admissions_data['application_timestamp'].dt.hour
admissions_data['acceptance_weekday'] = (admissions_data['acceptance_timestamp'].dt.dayofweek)+1
admissions_data['acceptance_month'] = admissions_data['acceptance_timestamp'].dt.month
admissions_data['acceptance_year'] = admissions_data['acceptance_timestamp'].dt.year
admissions_data['acceptance_hour'] = admissions_data['acceptance_timestamp'].dt.hour

#augment admissions data by :
    # extracting if application/offer was conditional/unconditional
    # if it was the students first choice or their insurance choice
    # indicating if the discipline codes differ between application and acceptance
    # if application campus differs from acceptance campus
admissions_data['firstchoice_application'] = np.where(admissions_data['application_admissions_code'].str[1:2] == 'F', 1, 0)
admissions_data['unconditional_application'] = np.where(admissions_data['application_admissions_code'].str[:1] == 'U', 1, 0)
admissions_data['discipline_code_disparity']=np.where(admissions_data['application_discipline_code'] == admissions_data['acceptance_discipline_code'], 1, 0)

#load student data, filter to relevant students, merge admissions data 
student_data=pd.read_csv('New Student Data.csv')
student_data=student_data[student_data['Alternative Offer Marker'] == 'Y']
student_data.rename(columns={'pre_nqa.Campus':'Campus_Chosen','Campus Group':'Campus_Given'},inplace=True)
combined_df=pd.merge(student_data,admissions_data,how='left',on='Student Code')
combined_df['Alternative or original campus'] = np.where(combined_df['Alternative or original campus'] == 'Studying at campus applied to ', 1, 0)
del(admissions_data,student_data)

#split new df into (dummy) features
X = pd.get_dummies(combined_df.drop(['application_year','acceptance_year','Count','Alternative Offer Marker',
                                      'Programme Code', 
                                      #'Current Sub-Discipline',#putting on seperate lines to make it easier to comment out during model tuning
                                      'application_discipline_code',
                                      'acceptance_discipline_code',
                                      'Student Code', 
                                      'Cohort Year', 'application_admissions_code','application_timestamp', 
                                      'acceptance_admissions_code','acceptance_timestamp'], axis=1))
#load trained model
model=load_model('alternative_offer_withdrawal_tf_prediction')

#apply trained model to new data
y_hat = model.predict(X)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
del(X)

#merge predicted withdrawal with admissions data to retrieve discarded columns
y_hat = pd.DataFrame(data=y_hat,columns=['Withdrawal Marker'])
admissions_data=pd.read_csv('New Admissions Data.csv')
admissions_data['min_nqa.DRL_IDRC']=admissions_data['acceptance_admissions_code']
admissions_data.rename(columns={'pre_nqa.DRL_IDRC':'application_admissions_code','min_nqa.DRL_IDRC':'acceptance_admissions_code','pre_nqa.drl_timestamp':'application_timestamp','min_nqa.drl_timestamp':'acceptance_timestamp','pre_nqa.DRL_MCRC':'application_discipline_code','min_nqa.DRL_MCRC':'acceptance_discipline_code'},inplace=True)
predicted_df=pd.merge(admissions_data,y_hat,left_index=True,right_index=True)
predicted_df.to_csv('./predicted_data/predicted_withdrawals.csv',index=False)
del(y_hat,admissions_data)

# =============================================================================
# # Prescriptive analysis (what steps to take)
# =============================================================================

combined_df=pd.read_csv('./predicted_data/predicted_withdrawals.csv')

#generate a list of IDs who are predicted to withdraw and send email containing these
predicted_withdrawal_id=combined_df[combined_df['Withdrawal Marker']==1]['Student Code']
predicted_withdrawal_id.to_csv('IDs of students predicted to withdraw.csv',index=False)
del(predicted_withdrawal_id)

#filter to only students predicted to withdraw
predicted_df=predicted_df[predicted_df['Withdrawal Marker']==1]

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

#generate an intro line to put in the email e.g. 'X/Y predicted withdrawals'
introtext=('Out of '+str(pd.DataFrame(combined_df['Withdrawal Marker'].value_counts()).iat[0,0])+' entries, '+str(pd.DataFrame(combined_df['Withdrawal Marker'].value_counts()).iat[1,0])+' are predicted to be at risk of withdrawing')

#generate a table of granularities and print how many predicted withdrawals are in each group at each level
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
#neaten up for printing
messagedata=messagedata.to_string(index=False)

#Anonymised for Github - uses my email and oauth2 json file
yag = yagmail.SMTP("", oauth2_file="")

#to can read from a csv file if multiple recipients are required
to = ''
subject = 'Withdrawal Prediction'
img1 = './predicted_graphs/Campus Group_predicted_withdrawals.png'
img2='./predicted_graphs/Current Sub-Discipline_predicted_withdrawals.png'
img3='./predicted_graphs/Current College_predicted_withdrawals.png'
idcodes='./predicted_data/IDs of students predicted to withdraw.csv'

body=f"""
Dear ,

Attached are student codes that have been flagged as being predicted to withdraw. Please contact Student Records for identification details

{introtext}
{messagedata}

Please see the attached graphs for more information    
    
Regards,
Nathan
"""

#send out email with above message, images, and student codes
yag.send(to = to, subject = subject, contents = body,attachments=[img1,img2,img3,idcodes])
