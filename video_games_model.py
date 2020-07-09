import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

'''
Please scroll to the bottom of the script for the voting classifier model!
'''

###############################################
#######                                 #######
#######    Data cleaning/engineering    #######
#######                                 #######
###############################################

# read the file
df = pd.read_csv('The directory containing the data')

# turn all release dates into years
years = []
i = 1990
while i < 2021:
    years.append(str(i))
    i += 1

j = 0
while j < len(df['ReleaseDate']):
    date = df.loc[j, 'ReleaseDate']
    date_split = date.split()
    if 2 <= len(date_split) <= 3 and date_split[-1] in years:
        year = date[-4:]
        year = int(year)
        df.loc[j, 'ReleaseDate'] = year
    elif '-2' in date:
        year = date[-2:]
        year = int(year) + 2000
        df.loc[j, 'ReleaseDate'] = year
    elif type(date) == int:
        pass
    else:
        df.loc[j, 'ReleaseDate'] = float('NaN')
    j += 1

df['ReleaseDate'] = df['ReleaseDate'].fillna(df['ReleaseDate'].mean())
df['ReleaseDate'] = df['ReleaseDate'].astype(int)
df['ReleaseDate'].hasnans

# drop duplicates and irrelevant features
df = df.drop_duplicates()
df = df.drop_duplicates() .T.drop_duplicates().T

df = df.drop(["QueryID","QueryName","ResponseName","DeveloperCount","PublisherCount","PCReqsHaveMin",
            "LinuxReqsHaveMin","MacReqsHaveMin","GenreIsNonGame","GenreIsIndie","GenreIsAction",
            "GenreIsAdventure","GenreIsCasual","GenreIsStrategy","GenreIsRPG","GenreIsEarlyAccess",
            "GenreIsSports","GenreIsRacing","GenreIsMassivelyMultiplayer","AboutText","Background",
            "ShortDescrip","DetailedDescrip","DRMNotice","ExtUserAcctNotice","HeaderImage","Reviews",
            "PCMinReqsText","PCRecReqsText","LinuxMinReqsText","LinuxRecReqsText","MacMinReqsText",
            "MacRecReqsText"],axis=1)

# encode True and False data points
df = df.replace(True, 1)
df = df.replace(False, 0)

# combining features
df["platform_quantity"] = df["PlatformWindows"] + df["PlatformLinux"] + df["PlatformMac"]
df = df.drop(["PlatformWindows","PlatformLinux","PlatformMac"], axis=1)

df['LegalNotice'] = np.where(df['LegalNotice'].str.len() > 2, 1, 0)
df['SupportURL'] = np.where(df['SupportURL'].str.len() > 2, 1, 0)
df['SupportEmail'] = np.where(df['SupportEmail'].str.len() > 2, 1, 0)
df['Website'] = np.where(df['Website'].str.len() > 2, 1, 0)

df["support_function"] = df['SupportURL'] + df['SupportEmail']
df = df.drop(['SupportURL','SupportEmail'], axis=1)

df["support_function"] = df["support_function"].replace(2, 1)

# replace supported languages data with number of languages the games support
language = ['Arabic','Bulgarian','Simplified Chinese','Traditional Chinese','Czech','Danish','Dutch',
            'English','Finnish','French','German','Greek','Hungarian','Italian','Japanese','Korean',
            'Norwegian','Polish','Portuguese','Portuguese-Brazil','Romanian','Russian','Spanish',
            'Swedish','Thai','Turkish','Ukrainian','Vietnamese']
df = df.reset_index(drop=True)
df["language"] = np.nan
for a in range(len(df)):
    counter = 0
    for lan in language:
        if lan in df["SupportedLanguages"][a]:
            counter += 1
    df["language"][a] = counter

df = df.drop(['SupportedLanguages'], axis=1)

# split data
df_target = df['IsFree']
df_base = df.drop(['IsFree'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_base, df_target, test_size = 0.2, random_state=5)

# cross-validation function
def crossValid(model, hyp, scores):
    for score in scores:
        print("# Tuning hyperparameters for %s" % score)
        print("\n")
        clf = GridSearchCV(model, hyp, cv=5,
                           scoring=score)
        clf.fit(X_train, Y_train)
        print("Best parameters set found on the training set:")
        print(clf.best_params_)
        print("\n")



###############################################
#######                                 #######
#######       Logistic Regression       #######
#######                                 #######
###############################################

# fit model
lr = LR()
lr_fit = lr.fit(X_train, Y_train)
train_score = lr.score(X_train, Y_train)
print(train_score)

# prediction
predicted = lr.predict(X_test)
test_score = lr.score(X_test, Y_test)
print(test_score)

# cross-validation
tuned_parameters = [{'penalty': ['l2'],
                     'C': [1, 10, 100],  'max_iter': [200, 300]},
                    {'penalty': ['l1'],
                     'C': [0.5, 1, 1.5, 10, 100], 'solver': ['liblinear'], 'max_iter': [200, 300]}]

scores = ['accuracy', 'f1_macro']
crossValid(lr, tuned_parameters, scores)



###############################################
#######                                 #######
#######    Support Vector Classifier    #######
#######                                 #######
###############################################

# fit model
svc = SVC()
svc_fit = svc.fit(X_train, Y_train)
train_score = svc.score(X_train, Y_train)
print(train_score)

# prediction
predicted = svc.predict(X_test)
test_score = svc.score(X_test, Y_test)
print(test_score)

# cross-validation
tuned_parameters = [{'kernel': ['sigmoid', 'rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 0.5, 1.5, 10, 100]}]

scores = ['accuracy', 'f1_macro']
crossValid(svc, tuned_parameters, scores)


###############################################
#######                                 #######
#######    Decision Trees Classifier    #######
#######                                 #######
###############################################

# fit model
dtc = DTC()
dtc_fit = dtc.fit(X_train, Y_train)
train_score = dtc.score(X_train, Y_train)
print(train_score)

# prediction
predicted = dtc.predict(X_test)
test_score = dtc.score(X_test, Y_test)
print(test_score)

# cross-validation
tuned_parameters = [{'criterion': ['gini', 'entropy'],
                     'max_depth': [3, 5, 7],
                     'min_samples_split': [3, 5, 7],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['accuracy', 'f1_macro']
crossValid(dtc, tuned_parameters, scores)


###############################################
#######                                 #######
#######             XGBoost             #######
#######                                 #######
###############################################

# fit model and predict
d_train = xgb.DMatrix(data=X_train, label=Y_train)
xgb_clf = xgb.XGBClassifier(n_jobs=6, objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.5, max_depth=3,
                            n_estimators=25)
xgb_clf.fit(X_train, Y_train)
train_score = xgb_clf.score(X_train, Y_train)
print(train_score)

# prediction
predicted = xgb_clf.predict(X_test)
test_score = xgb_clf.score(X_test, Y_test)
print(test_score)

# cross-validation
tuned_parameters = [{'n_estimators':[25, 50, 75], 'learning_rate':[0.25, 0.5, 0.75], 'colsample_bytree':[0.1, 0.3, 0.5]
                     , 'objective':['binary:logistic'], 'max_depth':[3, 5, 7]}]

scores = ['accuracy', 'f1_macro']
crossValid(xgb_clf, tuned_parameters, scores)

params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.5, 'max_depth': 3}
cv_results = xgb.cv(dtrain=d_train, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, as_pandas=True)
print(cv_results)


###############################################
#######                                 #######
#######        Voting Classifier        #######
#######                                 #######
###############################################

# reparameterise models with new hyperparameters
clf1 = LR(C=10, max_iter=200, penalty='l1', solver='liblinear')
clf2 = SVC(C=1, gamma=0.001, kernel='rbf', probability=True)
clf3 = DTC(criterion='gini', max_depth=3, max_features=None, min_samples_split=3)
clf4 = xgb.XGBClassifier(n_jobs=6, objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.5, max_depth=3,
                            n_estimators=25)

# fit model
eclf1 = VotingClassifier([('lr', clf1), ('svc', clf2), ('dtc', clf3), ('xgb', clf4)], voting='soft')
eclf1 = eclf1.fit(X_train, Y_train)
train_score = eclf1.score(X_train, Y_train)
print(train_score)

# prediction
predicted = eclf1.predict(X_test)
test_score = eclf1.score(X_test, Y_test)
print(test_score)