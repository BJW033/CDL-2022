#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:37:48 2022

@author: bwu
"""

import pandas as pd
import numpy as np
import joblib

# Import Data used for prediction
matches = pd.read_csv('match_scores_2022.csv')
players = pd.read_csv('player_data_2022.csv')

#===============================================================
# Seperate the match data by Team_A and Team_B, then stacks them on top
teams_A_2022 = matches[['Team_A','A_pts','game_id','Game_Mode','Team_A_Win','B_pts']]
teams_A_2022.columns = ['Team','pts','game_id','Game_Mode','Win','pts_allowed']

teams_B_2022 = matches[['Team_B','B_pts','game_id','Game_Mode','Team_A_Win','A_pts']]
teams_B_2022['Win'] = matches['Team_A_Win'].apply(lambda x: 0 if x==1 else 1)
teams_B_2022.drop('Team_A_Win',axis=1,inplace=True)
teams_B_2022.columns = ['Team','pts','game_id','Game_Mode','pts_allowed','Win']
teams_2022 = pd.concat([teams_A_2022,teams_B_2022])
teams_2022.reset_index(drop=True,inplace=True)
#===============================================================

#===============================================================
# Gets each team's average points in general and in the last five
# for each game mode
teams_2022.sort_values(['Team','Game_Mode','game_id'],ascending = [True,True,False],inplace=True)
teams_2022['Avg_pts']=teams_2022.groupby(['Team','Game_Mode'])['pts'].apply(lambda x: x.expanding().mean().shift())
teams_2022['Avg_L5'] = teams_2022.groupby(['Team','Game_Mode'])['pts'].transform(lambda x: x.rolling(5).mean().shift())

teams_2022['Win_Per']=teams_2022.groupby(['Team','Game_Mode'])['Win'].apply(lambda x: x.expanding().mean().shift())
teams_2022['Win_Per_L5'] = teams_2022.groupby(['Team','Game_Mode'])['Win'].transform(lambda x: x.rolling(5).mean().shift())

teams_2022['Avg_pts_allowed']=teams_2022.groupby(['Team','Game_Mode'])['pts_allowed'].apply(lambda x: x.expanding().mean().shift())
teams_2022['Avg_pts_allowed_L5'] = teams_2022.groupby(['Team','Game_Mode'])['pts_allowed'].transform(lambda x: x.rolling(5).mean().shift())
#===============================================================

#===============================================================
# Gets each team's average KD for each game mode
get_map_kd_2022 = pd.merge(players,matches[['game_id','Game_Mode']], on='game_id')
get_map_kd_2022.sort_values('game_id',ascending=False,inplace=True)
get_map_kd_2022 = get_map_kd_2022.groupby(['Team','game_id','Game_Mode'])[['Kills','Deaths']].sum()
get_map_kd_2022.reset_index(inplace=True)
get_map_kd_2022.sort_values('game_id',ascending=False,inplace=True)
get_map_kd_2022['Sum_Kills'] = get_map_kd_2022.groupby(['Team','Game_Mode'])['Kills'].apply(lambda x: x.expanding().sum().shift())
get_map_kd_2022['Sum_Deaths'] = get_map_kd_2022.groupby(['Team','Game_Mode'])['Deaths'].apply(lambda x: x.expanding().sum().shift())
get_map_kd_2022['Avg_KD'] = get_map_kd_2022['Sum_Kills']/get_map_kd_2022['Sum_Deaths']

# Duplicate Last 5 from before but with sum, then divide
get_map_kd_2022.sort_values('game_id',ascending=False,inplace=True)
get_map_kd_2022['Sum_Kills_L5'] = get_map_kd_2022.groupby(['Team','Game_Mode'])['Kills'].transform(lambda x: x.rolling(5).sum().shift())
get_map_kd_2022['Sum_Deaths_L5'] = get_map_kd_2022.groupby(['Team','Game_Mode'])['Deaths'].transform(lambda x: x.rolling(5).sum().shift())
get_map_kd_2022['Avg_KD_L5'] = get_map_kd_2022['Sum_Kills_L5']/get_map_kd_2022['Sum_Deaths_L5']
#===============================================================

def get_data(team, mode):
    """ Takes the team to look for and a game mode to get data for.
        Returns an array of the teams data in the order of:
            Avg_points, Avg_points_L5, Win_Per, Win_Per_L5, Avg_points_allowed,
            Avg_points,allowed_L5, Total_KD, Total_KD_L5"""
    ret = []
    filtered_DF = teams_2022[(teams_2022['Team'] == team) &
                             (teams_2022['Game_Mode'] == mode)].sort_values('game_id')
    filtered_KD_DF = get_map_kd_2022[(get_map_kd_2022['Team'] == team) &
                             (get_map_kd_2022['Game_Mode'] == mode)].sort_values('game_id')
    ret.append(filtered_DF['pts'].mean()) #AVG Points
    ret.append(filtered_DF['pts'].head(5).mean()) # AVG Points L5
    ret.append(filtered_DF['Win'].mean()) #Win Percentage
    ret.append(filtered_DF['Win'].head(5).mean()) #Win Percentage L5
    ret.append(filtered_DF['pts_allowed'].mean()) #AVG Points Allowed
    ret.append(filtered_DF['pts_allowed'].head(5).mean()) # AVG Points Allowed L5

    ret.append((filtered_KD_DF['Kills'].sum()/filtered_KD_DF['Deaths'].sum())) #Total KD
    ret.append((filtered_KD_DF['Kills'].head(5).sum()/filtered_KD_DF['Deaths'].head(5).sum())) #KD L5
    return ret

def get_matchup(team_A,team_B):
    """ Takes the names of two teams and returns DataFrame consisting of
        both teams averages for each of the three game modes"""
    matchup = []
    modes = ['Hardpoint','Search','Control']
    for m in modes:
        for t in [team_A,team_B]:
            start = [t,m]
            start.extend(get_data(t, m))
            filtered_DF = teams_2022[(teams_2022['Team'] == t) &
                             (teams_2022['Game_Mode'] == m)].sort_values('game_id')
            start.extend(filtered_DF['game_id'].shape)
            matchup.append(start)
            
    ret = pd.DataFrame(matchup)
    ret.columns = ['Team','Mode','Avg_pts','Avg_pts_L5',
                   'Win_per','Win_per_L5','Avg_pts_allowed',
                   'Avg_pts_allowed_L5','Total_KD','Total_KD_L5','Games_Played']
    return ret

matchup = get_matchup('Los Angeles Thieves', 'Toronto Ultra')


team_list = ['Atlanta FaZe','Boston Breach','Florida Mutineers',
                 'London Royal Ravens','Los Angeles Guerrillas','Los Angeles Thieves',
                 'Minnesota RØKKR','New York Subliners','OpTic Texas',
                 'Paris Legion','Seattle Surge','Toronto Ultra']

def get_all_avgs():
    """ Returns the average data for each team in each gamemode in a DataFrame"""
    avgs = []
    modes = ['Hardpoint','Search','Control']
    for m in modes:
        for t in team_list:
            start = [t,m]
            start.extend(get_data(t, m))
            filtered_DF = teams_2022[(teams_2022['Team'] == t) &
                             (teams_2022['Game_Mode'] == m)].sort_values('game_id')
            start.extend(filtered_DF['game_id'].shape)
            avgs.append(start)
    ret = pd.DataFrame(avgs)
    ret.columns = ['Team','Mode','Avg_pts','Avg_pts_L5',
                   'Win_per','Win_per_L5','Avg_pts_allowed',
                   'Avg_pts_allowed_L5','Total_KD','Total_KD_L5','Games_Played']

    return ret

all_avgs = get_all_avgs()



data = []
predictions = []
all_win_per = []
map_win_per = pd.DataFrame()
def chance_to_beat(team):
    """ Takes the name of a team as input. Calculates that teams chances
        of beating each of the other eleven teams. Models are imported
        and values are scaled to predict the chances of winning each 
        game mode individuals. Then the percentages are used to calculate
        series odds and odds in general. Nothing is outputed, global
        variable map_win_per is modified"""
    print('######################################################')
    
    chances = []
    for t in team_list:
        if t==team:
            continue
        modes = ['Hardpoint','Search','Control']
        scales = ["./HP_scaler.joblib","./S_scaler.joblib","./C_scaler.joblib"]
        models = ["./HP_random_forest.joblib","./S_Logistic.joblib","./C_logistic.joblib"]
        it = 0
        data = []
        predictions = []
        c = []
        for m in modes:
           
            predict = get_data(team,m)
            predict.extend(get_data(t,m))
            predict = np.array(predict)
            data.append(predict[0:8])
            data[len(data)-1] = np.append(data[len(data)-1],team)
            data[len(data)-1] = np.append(data[len(data)-1],m)
            
            data.append(predict[8:])
            data[len(data)-1] = np.append(data[len(data)-1],t)
            data[len(data)-1] = np.append(data[len(data)-1],m)
            
            scale = joblib.load(scales[it])
            model = joblib.load(models[it])
            #print(model.predict_proba(scale.transform(predict.reshape(1,-1))))
            model_predict = model.predict_proba(scale.transform(predict.reshape(1,-1)))
            A_win = model_predict[0][1]
            B_win = model_predict[0][0]
            model_prediction = [m,A_win,B_win]
            predictions.append(model_prediction)
            it = it + 1
        
        percentages = pd.DataFrame(predictions)
        percentages.columns = ['Mode',team,t]
        global map_win_per
        map_win_per = pd.concat([map_win_per,percentages])
        
        data_DF = pd.DataFrame(data)
        
        data_DF.columns = ['Avg_pts','Avg_pts_L5',
          'Win_Per','Win_Per_L5','Avg_pts_allowed',
          'Avg_pts_allowed_L5','Avg_KD','Avg_KD_L5',
          'Team','Mode']
        
        data_DF = data_DF[['Team','Mode','Avg_pts','Avg_pts_L5',
          'Win_Per','Win_Per_L5','Avg_pts_allowed',
          'Avg_pts_allowed_L5','Avg_KD','Avg_KD_L5']]
        
        series_Scores = [[team, t],[t,team]]
        series_Scores[0].append(percentages[team].product())
        series_Scores[1].append(percentages[t].product())
        
        three_one = [predictions[0][2]*predictions[1][1]*predictions[2][1]*predictions[0][1]+
                     predictions[0][1]*predictions[1][2]*predictions[2][1]*predictions[0][1]+
                     predictions[0][1]*predictions[1][1]*predictions[2][2]*predictions[0][1],
                     predictions[0][1]*predictions[1][2]*predictions[2][2]*predictions[0][2]+
                     predictions[0][2]*predictions[1][1]*predictions[2][2]*predictions[0][2]+
                     predictions[0][2]*predictions[1][2]*predictions[2][1]*predictions[0][2]]
        
        series_Scores[0].append(three_one[0])
        series_Scores[1].append(three_one[1])
        
        three_two = [predictions[0][1]*predictions[1][1]*predictions[2][2]*predictions[0][2]*predictions[1][1]+
                     predictions[0][1]*predictions[1][2]*predictions[2][1]*predictions[0][2]*predictions[1][1]+
                     predictions[0][2]*predictions[1][1]*predictions[2][1]*predictions[0][2]*predictions[1][1]+
                     predictions[0][2]*predictions[1][1]*predictions[2][2]*predictions[0][1]*predictions[1][1]+
                     predictions[0][2]*predictions[1][2]*predictions[2][1]*predictions[0][1]*predictions[1][1]+
                     predictions[0][1]*predictions[1][2]*predictions[2][2]*predictions[0][1]*predictions[1][1],
                     predictions[0][2]*predictions[1][2]*predictions[2][1]*predictions[0][1]*predictions[1][2]+
                     predictions[0][2]*predictions[1][1]*predictions[2][2]*predictions[0][1]*predictions[1][2]+
                     predictions[0][1]*predictions[1][2]*predictions[2][2]*predictions[0][1]*predictions[1][2]+
                     predictions[0][1]*predictions[1][2]*predictions[2][1]*predictions[0][2]*predictions[1][2]+
                     predictions[0][1]*predictions[1][1]*predictions[2][2]*predictions[0][2]*predictions[1][2]+
                     predictions[0][2]*predictions[1][1]*predictions[2][1]*predictions[0][2]*predictions[1][2]]
        
        series_Scores[0].append(three_two[0])
        series_Scores[1].append(three_two[1])
        
        Final_Prediction = pd.DataFrame(series_Scores)
        Final_Prediction.columns = ['Team A','Team B','3-0','3-1','3-2']
        Final_Prediction['Win_Overall'] = Final_Prediction['3-0'] + Final_Prediction['3-1'] + Final_Prediction['3-2']
        
        all_win_per.append(series_Scores[0])
        all_win_per.append(series_Scores[1])
        
        Final_Prediction['3-0'] = (round((Final_Prediction['3-0'] * 100),3)).astype(str) + '%'
        Final_Prediction['3-1'] = (round((Final_Prediction['3-1'] * 100),3)).astype(str) + '%'
        Final_Prediction['3-2'] = (round((Final_Prediction['3-2'] * 100),3)).astype(str) + '%'
        Final_Prediction['Win_Overall'] = (round((Final_Prediction['Win_Overall'] * 100),3)).astype(str) + '%'
        # print(percentages)
        # print()
        # print(Final_Prediction)
        # print('######################################################')
        c = [team, t,Final_Prediction.iloc[1,0],Final_Prediction.iloc[1,1],Final_Prediction.iloc[1,2],Final_Prediction.iloc[1,3]]
        chances.append(c)
        
    chances = pd.DataFrame(chances)
    chances.columns=['Team A','Team B','3-0','3-1','3-2','Win_Overall']
    chances.sort_values('Win_Overall',ascending = False, inplace=True)
    #print("Chances to Beat " + team)
   # print(chances)
   
# Calculate each teams chances to beat the others
for t in team_list:
    chance_to_beat(t)



# Covert the all_win_per to DataFrame
all_win_per_DF = pd.DataFrame(all_win_per)
all_win_per_DF['Win_Ovearll'] = all_win_per_DF[2]+all_win_per_DF[3]+all_win_per_DF[4]
all_win_per_DF.columns = ['Team A','Team B','3-0','3-1','3-2','Win_Overall']

# Each match up of two teams has two observations in all_win_per_DF where each team
# is Team_A and Team_B. We average the chances at each series count.
avg_outcomes = all_win_per_DF.groupby(['Team A','Team B']).mean()
avg_outcomes.reset_index(inplace=True)


def get_outcome(teamA,teamB):
    """ Takes two teams to calculate the estimated series count. 
    Prints the chances of each team winning each individual game mode.
    Returns a DataFrame of the projected series counts. 
    
    """

    print(map_win_per[(map_win_per[teamA].notnull()) &
        (map_win_per[teamB].notnull())][['Mode',teamA,teamB]].groupby('Mode').mean())
    print("######################################################")
    return avg_outcomes[((avg_outcomes['Team A'] == teamA) &
                     (avg_outcomes['Team B'] == teamB)) |
                      ((avg_outcomes['Team A'] == teamB) &
                     (avg_outcomes['Team B'] == teamA))]

M1 = get_outcome('Atlanta FaZe','Florida Mutineers')
M2 = get_outcome('Florida Mutineers','London Royal Ravens')
M3 = get_outcome('OpTic Texas','Los Angeles Guerrillas')
M4 = get_outcome('Los Angeles Thieves','New York Subliners')





#Creates excel sheet of all teams averages
#all_avgs.to_excel("Team Averages - Major 1 Qualifiers.xlsx",index=False)

