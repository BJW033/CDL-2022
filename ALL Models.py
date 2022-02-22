#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:43:28 2022

@author: bwu
"""

import pandas as pd
import random 
import joblib


import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



# Import the data from matches and players.
# As of now, data from 2021 is used to fit models, 2022 data is used to predict
all_maps = pd.read_csv('match_scores.csv')
all_players = pd.read_csv('player_data.csv')



#===============================================================================
#Sererate map data by teams
teams_A = all_maps[['Team_A','A_pts','game_id','Game_Mode','Team_A_Win','B_pts']]
teams_A.columns = ['Team','pts','game_id','Game_Mode','Win','pts_allowed']

teams_B = all_maps[['Team_B','B_pts','game_id','Game_Mode','Team_A_Win','A_pts']]
teams_B['Win'] = all_maps['Team_A_Win'].apply(lambda x: 0 if x==1 else 1)
teams_B.drop('Team_A_Win',axis=1,inplace=True)
teams_B.columns = ['Team','pts','game_id','Game_Mode','pts_allowed','Win']
teams = pd.concat([teams_A,teams_B])
teams.reset_index(drop=True,inplace=True)

HP_map_DF= all_maps[all_maps['Game_Mode']=="Hardpoint"]
S_map_DF= all_maps[all_maps['Game_Mode']=="Search"]
C_map_DF= all_maps[all_maps['Game_Mode']=="Control"]
DFs = [HP_map_DF,S_map_DF,C_map_DF]

# These models are the most consistant in accuracy determined
# after fitting the data of each gamemode to a variety of different models.
from sklearn.ensemble import RandomForestClassifier
HP_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
from sklearn.linear_model import LogisticRegression
S_classifier = LogisticRegression(random_state = 0)
C_classifier = LogisticRegression(random_state = 0)
models = [HP_classifier,S_classifier,C_classifier]

labels = ['HP','S','C']
Summary_Stats = []
model_index = 0


#===============================================================================
#Randomly assign each map a coin flip to evenly distribute A and B winners
for df in DFs:  
    #print('#########')
    #print(df['Team_A_Win'].sum())
    index = 0
    dup=df
    for index in df.index:
        flip = random.randint(0, 1)
        df.loc[index,'PlaceHolder'] = flip
        dup.loc[index,'PlaceHolder'] = abs(flip-1)
        
    df[['Team_A','Team_B']] = df[['Team_B','Team_A']].where(df['PlaceHolder'] == 1, df[['Team_A','Team_B']].values)
    df[['A_pts','B_pts']] = df[['B_pts','A_pts']].where(df['PlaceHolder'] == 1, df[['A_pts','B_pts']].values)
    #testDF['Team_A_Win']] = testDF[['Team_A_Win'].where(testDF['PlaceHolder'] == 1, testDF[['A_pts','B_pts']].values)
    df['Team_B_Win'] = df['Team_A_Win'].apply(lambda x: 0 if x==1 else 1)
    df[['Team_A_Win','Team_B_Win']] = df[['Team_B_Win','Team_A_Win']].where(df['PlaceHolder'] == 1, df[['Team_A_Win','Team_B_Win']].values)
    df.drop(['Team_B_Win','PlaceHolder'],axis=1,inplace=True)
    #print(df['Team_A_Win'].sum())


#===============================================================================
#Averages over the Last 5 and General
teams.sort_values(['Team','Game_Mode','game_id'],ascending = [True,True,False],inplace=True)
teams['Avg_pts']=teams.groupby(['Team','Game_Mode'])['pts'].apply(lambda x: x.expanding().mean().shift())
teams['Avg_L5'] = teams.groupby(['Team','Game_Mode'])['pts'].transform(lambda x: x.rolling(5).mean().shift())

teams['Win_Per']=teams.groupby(['Team','Game_Mode'])['Win'].apply(lambda x: x.expanding().mean().shift())
teams['Win_Per_L5'] = teams.groupby(['Team','Game_Mode'])['Win'].transform(lambda x: x.rolling(5).mean().shift())

teams['Avg_pts_allowed']=teams.groupby(['Team','Game_Mode'])['pts_allowed'].apply(lambda x: x.expanding().mean().shift())
teams['Avg_pts_allowed_L5'] = teams.groupby(['Team','Game_Mode'])['pts_allowed'].transform(lambda x: x.rolling(5).mean().shift())


get_map_kd = pd.merge(all_players,all_maps[['game_id','Game_Mode']], on='game_id')
get_map_kd.sort_values('game_id',ascending=False,inplace=True)
get_map_kd=get_map_kd.groupby(['Team','game_id','Game_Mode'])[['Kills','Deaths']].sum()
get_map_kd.reset_index(inplace=True)
get_map_kd.sort_values('game_id',ascending=False,inplace=True)
get_map_kd['Sum_Kills'] = get_map_kd.groupby(['Team','Game_Mode'])['Kills'].apply(lambda x: x.expanding().sum().shift())
get_map_kd['Sum_Deaths'] = get_map_kd.groupby(['Team','Game_Mode'])['Deaths'].apply(lambda x: x.expanding().sum().shift())
get_map_kd['Avg_KD'] = get_map_kd['Sum_Kills']/get_map_kd['Sum_Deaths']

# Duplicate Last 5 from before but with sum, then divide
get_map_kd.sort_values('game_id',ascending=False,inplace=True)
get_map_kd['Sum_Kills_L5'] = get_map_kd.groupby(['Team','Game_Mode'])['Kills'].transform(lambda x: x.rolling(5).sum().shift())
get_map_kd['Sum_Deaths_L5'] = get_map_kd.groupby(['Team','Game_Mode'])['Deaths'].transform(lambda x: x.rolling(5).sum().shift())
get_map_kd['Avg_KD_L5'] = get_map_kd['Sum_Kills_L5']/get_map_kd['Sum_Deaths_L5']


#===============================================================================
#Merge Total Average Pts and WinPer
for df in DFs:
    df = pd.merge(df, teams[['Team','game_id','Avg_pts','Win_Per','Avg_pts_allowed','Avg_pts_allowed_L5']],  how='left', left_on=['Team_A','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
       
    df = pd.merge(df, teams[['Team','game_id','Avg_pts','Win_Per','Avg_pts_allowed','Avg_pts_allowed_L5']],  how='left', left_on=['Team_B','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
         
    
    #Merge Average Last 5 Pts and WinPer
    df = pd.merge(df, teams[['Team','game_id','Avg_L5','Win_Per_L5']],  how='left', left_on=['Team_A','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
       
    df = pd.merge(df, teams[['Team','game_id','Avg_L5','Win_Per_L5']],  how='left', left_on=['Team_B','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
    
    #Merge Total Average Kds
    df = pd.merge(df, get_map_kd[['Team','game_id','Avg_KD']],  how='left', left_on=['Team_A','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
       
    df = pd.merge(df, get_map_kd[['Team','game_id','Avg_KD']],  how='left', left_on=['Team_B','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
    
    #Merge Average Last 5 Kds
    df = pd.merge(df, get_map_kd[['Team','game_id','Avg_KD_L5']],  how='left', left_on=['Team_A','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
       
    df = pd.merge(df, get_map_kd[['Team','game_id','Avg_KD_L5']],  how='left', left_on=['Team_B','game_id'], right_on = ['Team','game_id'])    
    df.drop('Team',axis=1,inplace=True)
#===============================================================================
    df.drop(['A_pts','B_pts','Date'],axis=1,inplace=True)
    
    df.columns = ['Team_A', 'Team_B', 'game_id', 
                      'Game_Mode','Team_A_Win','A_avg_pts',
                      'A_Win_Per','A_avg_pts_allowed','A_avg_pts_allowed_L5',
                      'B_avg_pts','B_Win_Per','B_avg_pts_allowed','B_avg_pts_allowed_L5',
                      'A_avg_pts_L5','A_Win_Per_L5','B_avg_pts_L5',
                      'B_Win_Per_L5','A_avg_KD','B_avg_KD','A_avg_KD_L5',
                      'B_avg_KD_L5']
    df = df[['Team_A', 'Team_B', 'game_id', 
             'Game_Mode','Team_A_Win','A_avg_pts','A_avg_pts_L5',
             'A_Win_Per','A_Win_Per_L5','A_avg_pts_allowed',
             'A_avg_pts_allowed_L5','A_avg_KD',
             'A_avg_KD_L5',
             'B_avg_pts','B_avg_pts_L5','B_Win_Per','B_Win_Per_L5',
             'B_avg_pts_allowed','B_avg_pts_allowed_L5',
             'B_avg_KD','B_avg_KD_L5']]
    
    # Drop observations consisting of NA values
    df.dropna(inplace=True)
    
    # Remove non-numeric values and game_id
    df.drop(['Team_A','Team_B','game_id','Game_Mode'],inplace=True,axis=1)
    
    
    # Seperate the dependent variable (Team_A_Win) from the independent variables
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Train/Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Scale (all the models that are being used require scaling)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # UNCOMMENT THE COMMENTS IN THIS STATEMENT IN ORDER TO SAVE THE MODELS
    # AND THE SCALES IN ORDER TO IMPORT THEM IN ANOTHER FILE
    if model_index == 0:
        print("fitting hardpoint data")
        mod='Forest for Hardpoint' 
        models[model_index].fit(X_train, y_train)
        #joblib.dump(models[model_index], "./HP_random_forest.joblib")
        #joblib.dump(sc, "./HP_scaler.joblib")
    elif model_index == 1:
        print("fitting search data")
        mod='Logistic For Search' 
        models[model_index].fit(X_train, y_train)
        #joblib.dump(models[model_index], "./S_Logistic.joblib")
        #joblib.dump(sc, "./S_scaler.joblib")
    elif model_index == 2:
        print("fitting control data")
        mod='Logistic For Control' 
        models[model_index].fit(X_train, y_train)
        #joblib.dump(models[model_index], "./C_Logistic.joblib")
        #joblib.dump(sc, "./C_scaler.joblib")
        
        
    # Print accuracy results   
    from sklearn.metrics import confusion_matrix, accuracy_score
    y_pred = models[model_index].predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(mod)
    print(cm)
    print(accuracy_score(y_test, y_pred))
    Summary_Stats.append([labels[model_index],accuracy_score(y_test, y_pred)])
    model_index = model_index + 1
    
SS_DF = pd.DataFrame(Summary_Stats)   
SS_DF.columns=['Mode','Accuracy Score']

print(SS_DF)




