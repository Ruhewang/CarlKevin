#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os
import numpy as np
import pandas as pd
import datetime

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers


# In[55]:


raw_raw_table = pd.read_csv("./exported_csv/nn_table.csv")

# answer = pd.read_csv("cx_answers.csv")


# In[56]:


raw_table = raw_raw_table
raw_table = raw_table[raw_table.FINAL_MEAL != 0]
raw_table['weekday'] = pd.DatetimeIndex(raw_table['DEP_DATE']).dayofweek +1
#raw_table = raw_table[(raw_table.SEAT_CLASS != "A") & (raw_table.SEAT_CLASS != "T")]
raw_table = raw_table[(raw_table.SEAT_CLASS == "Y")]
raw_table['booking_rate'] = raw_table.BOOKED_QUANTITY/raw_table.CAPACITY_QUANTITY
raw_table['rollback_logtime'] = (pd.to_datetime(raw_table.DEP_DATE)-pd.to_datetime(raw_table.LOG_TIME)).dt.days

# answer_day=answer[(answer.FLIGHT_DATE >= "2014-08-01") & (answer.FLIGHT_DATE <= "2019-05-31")]
# train_answer_day =answer[(answer.FLIGHT_DATE >= "2014-08-01") & (answer.FLIGHT_DATE <= "2019-04-30")]
# test_answer_day = answer[(answer.FLIGHT_DATE > "2019-04-30") & (answer.FLIGHT_DATE <= "2019-05-31")]
# train_answer_day = train_answer_day[["FLIGHT_DATE","CX_MEAL_day"]]
# test_answer_day = test_answer_day[["FLIGHT_DATE","CX_MEAL_day"]]


# In[57]:


# Exclude typhoon days
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2018-09-15"].index)
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2018-09-16"].index)
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2018-09-17"].index)
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2017-08-22"].index)
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2018-08-23"].index)
raw_table = raw_table.drop(raw_table[raw_table.DEP_DATE == "2018-08-24"].index)


# In[58]:


raw_table['year'] = pd.DatetimeIndex(raw_table['DEP_DATE']).year
raw_table['month'] = pd.DatetimeIndex(raw_table['DEP_DATE']).month
raw_table['day'] = pd.DatetimeIndex(raw_table['DEP_DATE']).day


# In[59]:


list(raw_table)


# In[60]:


raw_table2.shape


# In[61]:


# Add weighted book
BookedWeight = pd.read_csv("./BookQTY_weight.csv")
BookedWeight["year"] = np.int64([k[0:4] for k in BookedWeight.month])
BookedWeight["month"] = np.int64([k[6:8] for k in BookedWeight.month])
raw_table = raw_table.merge(BookedWeight,on= ["year","month"])
raw_table["weighted_BOOKED_QUANTITY"] = raw_table.weight * raw_table.BOOKED_QUANTITY
raw_table["weighted_BOOKED_MEAL"] = raw_table.weight * raw_table.BOOKED_MEAL


# In[62]:


def prepare_holidays(raw_table):
    def getHolidays(row):
        # Christmas
        if row['month'] == 12 and row['day'] == 25:
            return 1
        # Labor Day
        elif row['month'] == 5 and row['day'] == 1:
            return 1
        # National Independence
        elif row['month'] == 10 and row['day'] == 1:
            return 1
        # Spring Festival (Jan)
        elif row['month'] == 1 and 24 <= row['day'] <= 31:
            return 1
        # Spring Festival (Feb)
        elif row['month'] == 2 and 1 <= row['day'] <= 20:
            return 1
        return 0

    def getHolidayRange(row):
        # Christmas
        if row['month'] == 12 and 18 <= row['day'] <= 31:
            return 1
        # Labor Day (April)
        elif row['month'] == 4 and 23 <= row['day'] <= 30:
            return 1
        #Labor Day (May)
        elif row['month'] == 5 and 1 <= row['day'] <= 7:
            return 1
        # National Independence (Sept)
        elif row['month'] == 10 and 23 <= row['day'] <= 30:
            return 1
        # National Independence (Oct)
        elif row['month'] == 10 and 1 <= row['day'] <= 7:
            return 1
        # Spring Festival (Jan)
        elif row['month'] == 1 and 24 <= row['day'] <= 31:
            return 1
        # Spring Festival (Feb)
        elif row['month'] == 2 and 1 <= row['day'] <= 20:
            return 1
        return 0

    raw_table['isPublicHoliday'] = raw_table.apply(lambda row: getHolidays(row), axis=1)
    raw_table['isHolidayRange'] = raw_table.apply(lambda row: getHolidayRange(row), axis=1)


# In[63]:


def prepare_holidays_one_hot(raw_table):
    def springFestival(row):
        # Spring Festival (Jan)
        if row['month'] == 1 and 24 <= row['day'] <= 31:
            return 1
        # Spring Festival (Feb)
        elif row['month'] == 2 and 1 <= row['day'] <= 20:
            return 1
        return 0

    def christmas(row):
        # Christmas
        if row['month'] == 12 and 18 <= row['day'] <= 31:
            return 1
        return 0

    def laborDay(row):
        # Labor Day (April)
        if row['month'] == 4 and 23 <= row['day'] <= 30:
            return 1
        #Labor Day (May)
        elif row['month'] == 5 and 1 <= row['day'] <= 7:
            return 1
        return 0

    def independence(row):    
        # National Independence (Sept)
        if row['month'] == 9 and 23 <= row['day'] <= 30:
            return 1
        # National Independence (Oct)
        elif row['month'] == 10 and 1 <= row['day'] <= 7:
            return 1
        return 0

    raw_table['Spring_Festival'] = raw_table.apply(lambda row: springFestival(row), axis=1)
    raw_table['Christmas'] = raw_table.apply(lambda row: christmas(row), axis=1)
    raw_table['Labor_Day'] = raw_table.apply(lambda row: laborDay(row), axis=1)
    raw_table['Independence_Day'] = raw_table.apply(lambda row: independence(row), axis=1)


# In[64]:


def prepare_departure_quarterly(raw_table):
    def midnight(row):
        if 0 <= row['FLIGHT_TIME'] <= 600:
            return 1
        return 0

    def morning(row):
        if 601 <= row['FLIGHT_TIME'] <= 1200:
            return 1
        return 0

    def afternoon(row):
        if 1201 <= row['FLIGHT_TIME'] <= 1800:
            return 1
        return 0

    def evening(row):
        if 1801 <= row['FLIGHT_TIME'] <= 2359:
            return 1
        return 0

    raw_table['Midnight'] = raw_table.apply(lambda row: midnight(row), axis=1)
    raw_table['Morning'] = raw_table.apply(lambda row: morning(row), axis=1)
    raw_table['Afternoon'] = raw_table.apply(lambda row: afternoon(row), axis=1)
    raw_table['Evening'] = raw_table.apply(lambda row: evening(row), axis=1)


# In[65]:


def prepare_distances(raw_table):
    sector_list = pd.read_csv("SECTOR_List.csv")

    def distance(row):
        if row["TO_SECTOR"] == "HKG":
            return sector_list.loc[sector_list['SECTOR'] == row["FROM_SECTOR"], 'Distance'].iloc[0]
        else:
            return sector_list.loc[sector_list['SECTOR'] == row["TO_SECTOR"], 'Distance'].iloc[0]

    raw_table['Distance'] = raw_table.apply(lambda row: distance(row), axis=1)
    distance_one_hot = pd.get_dummies(raw_table['Distance'], prefix="distance")
    raw_table = raw_table.join(distance_one_hot)


# In[66]:


# Potentially long process!
# Comment out data fields which won't be used to speed up
prepare_holidays(raw_table)
prepare_holidays_one_hot(raw_table)
prepare_departure_quarterly(raw_table)
prepare_distances(raw_table)


# In[67]:


total_fight_detail = raw_table[["DEP_DATE","FLIGHT_NO","TO_SECTOR","FLIGHT_TIME","SEAT_CLASS",'SECTOR_NO',"PROS_FORECAST","FORECAST_MEAL","weekday","numOfWeek","Freqencyweekly","isPublicHoliday","isHolidayRange","Midnight","Morning", "Afternoon", "Evening", "Distance"]]
training_fight_detail = total_fight_detail[(total_fight_detail.DEP_DATE >= "2014-08-01") & (total_fight_detail.DEP_DATE <= "2019-04-30")]
testing_fight_detail = total_fight_detail[(total_fight_detail.DEP_DATE > "2019-04-30") & (total_fight_detail.DEP_DATE <= "2019-05-31")]


# In[68]:


flight_one_hot = pd.get_dummies(raw_table['FLIGHT_NO'],prefix="flightNo")
#weekday_one_hot = pd.get_dummies(raw_table['weekday'],prefix="weekday")
# year_one_hot = pd.get_dummies(raw_table['year'],prefix="year")
# month_one_hot = pd.get_dummies(raw_table['month'],prefix="month")
# day_one_hot = pd.get_dummies(raw_table['day'],prefix="day")
sector_one_hot = pd.get_dummies(raw_table['TO_SECTOR'],prefix="sectorCode")
seat_one_hot = pd.get_dummies(raw_table['SEAT_CLASS'],prefix="seat")
distance_one_hot = pd.get_dummies(raw_table['Distance'], prefix="distance")


# In[69]:


# raw_table = raw_table.join(seat_one_hot)
raw_table = raw_table.join(distance_one_hot)
raw_table = raw_table.join(sector_one_hot)
raw_table = raw_table.join(flight_one_hot)


# In[70]:


# For output purposes only
default = ['MAX_SALE_QUANTITY', 'CAPACITY_QUANTITY', 'FORECAST_QUANTITY', 'BOOKED_QUANTITY', 'NO_MEAL_SERVE',
           'year', 'month', 'day']
to_drop = list(raw_table)


# In[71]:


class Experiment:
    def __init__(self, exp_no, variables):
            self._exp_no = exp_no
            self._variables = variables
            s = set(self._variables)
            self._drop_vars = [x for x in to_drop if x not in s]

        
    def getVariables(self):
        return self._variables
        
    def getDropVariables(self):
        return self._drop_vars
    
    def getExpNo(self):
        return self._exp_no


# In[72]:


# Shortcuts
Baseline = ['DEP_DATE','FORECAST_MEAL','FINAL_MEAL']
base = ['DEP_DATE', 'MAX_SALE_QUANTITY', 'CAPACITY_QUANTITY', 'FORECAST_QUANTITY', 'BOOKED_QUANTITY', 
        'NO_MEAL_SERVE', 'FINAL_MEAL',
        'year', 'month', 'day']
exp2 = ['DEP_DATE', 'MAX_SALE_MEAL', 'CAPACITY_MEAL', 'FORECAST_MEAL', 'BOOKED_MEAL', 
        'NO_MEAL_SERVE', 'FINAL_MEAL',
        'year', 'month', 'day']
holidays = ["Spring_Festival", "Christmas", "Labor_Day", "Independence_Day"]
departs = ["Midnight", "Morning", "Afternoon", "Evening"]
distances = ["distance_Long", "distance_Mid", "distance_Short"]
FLIGHT_NO_1hot= [k for k in list(raw_table) if 'flightNo_' in k]
SECTOR_1hot= [k for k in list(raw_table) if 'sectorCode_' in k]

# Add list of desired variables to test
experiments = [
#   Experiment("000", Baseline)
#   Experiment("001", base),
#   Experiment("002", exp2),
#   Experiment("003", base + ["weekday"]),
#   Experiment("004", base + FLIGHT_NO_1hot),
#   Experiment("005", base + SECTOR_1hot),
#   Experiment("006", base + ["numOfWeek"]),
#   Experiment("007", base + ["Freqencyweekly"]),
#   Experiment("008", base + ["isPublicHoliday"]),
#   Experiment("009", base + ["isHolidayRange"]),
#   Experiment("010", base + holidays),
#   Experiment("011", base + departs),
#   Experiment("012", base + distances),
  Experiment("013", base + ["weighted_BOOKED_QUANTITY"]),
#   Experiment("014", base + ["booking_rate"]),
#   Experiment("015", base + ["rollback_logtime"])
]


# In[73]:


for exp in experiments:
    test_flight_mae_list = []
    test_day_mae_list = []

    test_flight_sd_list = []
    test_day_sd_list = []
    
    best_mae = 1
    
    # Drop features
    model_table = raw_table.copy()
    for feature in exp.getDropVariables():
        model_table = model_table.drop(feature, axis = 1)
    
    print(list(model_table))
    
    train_start = "2014-08-01"
    train_end ="2019-04-30"
    test_start = "2019-04-30"
    test_end = "2019-05-31"
    
    detail_directory = "experiments/" + exp.getExpNo()
    if not os.path.exists(detail_directory):
        os.mkdir(detail_directory)
        
    txt_path = os.path.join(detail_directory, 'results.txt')
    with open(txt_path, "w") as text_file:
        text_file.write("Experiment      : {}\n".format(exp.getExpNo()))
        text_file.write("vars            : {}\n".format(exp.getVariables()))
        text_file.write("Train date      : {} to {}\n".format(train_start,train_end))
        text_file.write("Test date       : {} to {}\n".format(test_start,test_end))
        text_file.write("-" * 50)
        text_file.write("\n")
    
    # Do average over n runs
    for i in range(5):
  
        # Build model
        training_set = model_table[(model_table.DEP_DATE >= train_start) & (model_table.DEP_DATE <= train_end)]
        test_set     = model_table[(model_table.DEP_DATE > test_start) & (model_table.DEP_DATE <= test_end)]
        DEP_DATE_training = training_set.pop('DEP_DATE')
        DEP_DATE_test = test_set.pop('DEP_DATE')
                
        y_train=training_set.pop("FINAL_MEAL")
        X_train=training_set
        y_test=test_set.pop("FINAL_MEAL")
        X_test=test_set
        
        if exp.getExpNo() == "000":
            test_result_table = pd.DataFrame({'prediction':X_test.FORECAST_MEAL, 'truth':y_test})
            test_result_table = pd.concat([testing_fight_detail.reset_index(drop=True),test_result_table],axis=1)
            train_result_table = pd.DataFrame({'prediction':X_train.FORECAST_MEAL, 'truth':y_train})
            train_result_table = pd.concat([training_fight_detail.reset_index(drop=True),train_result_table],axis=1)
            break
        
        print("Run #" + str(i+1)),
        


        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        y_train = y_train.values
        y_test = y_test.values

        model = tf.keras.Sequential()
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))

        model.compile(optimizer=tf.train.AdamOptimizer(0.1),
                      loss='mse',
                      metrics=['mae'])

        model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=0)

        y_pred = model.predict(X_test)
        X_pred = model.predict(X_train)
        X_pred = X_pred.reshape(-1)
        y_pred = y_pred.reshape(-1)

        test_result_table = pd.DataFrame({'prediction':y_pred, 'truth':y_test})
        test_result_table = pd.concat([testing_fight_detail.reset_index(drop=True),test_result_table],axis=1)

        train_result_table = pd.DataFrame({'prediction':X_pred, 'truth':y_train})
        train_result_table = pd.concat([training_fight_detail.reset_index(drop=True),train_result_table],axis=1)

        # Note: This table is not used
        total_result_table = train_result_table.append(pd.DataFrame(data = test_result_table), ignore_index=True)
        
        run_mae = mean_absolute_error(test_result_table.prediction,test_result_table.truth)/np.mean(test_result_table.truth)
        if run_mae >= 0.5:
            continue
    
        test_flight_mae_list.append(run_mae)
        test_flight_sd_list.append(np.sqrt(np.var((test_result_table.prediction - test_result_table.truth)/test_result_table.truth)))
        
        grouped_test = test_result_table.groupby('DEP_DATE').agg({'prediction':'sum', 'truth': 'sum'})
        grouped_train = train_result_table.groupby('DEP_DATE').agg({'prediction':'sum', 'truth': 'sum'})
        
        test_day_mae_list.append(mean_absolute_error(grouped_test.prediction, grouped_test.truth)/np.mean(grouped_test.truth))
        test_day_sd_list.append(np.sqrt(np.var((grouped_test.prediction - grouped_test.truth)/grouped_test.truth)))
        
        if run_mae < best_mae:
            best_test_table = test_result_table
            best_train_table = train_result_table

    grouped_test = best_test_table.groupby(['DEP_DATE','SEAT_CLASS',"weekday","numOfWeek","isPublicHoliday","isHolidayRange"]).agg(
        {'prediction':'sum', 'truth': 'sum'}
    )
    grouped_train = best_train_table.groupby(['DEP_DATE','SEAT_CLASS',"weekday","numOfWeek","isPublicHoliday","isHolidayRange"]).agg(
        {'prediction':'sum', 'truth': 'sum'}
    )

    # Calculate absolute difference
    best_test_table['absdiff'] = abs(best_test_table['prediction'] - best_test_table['truth'])
    best_test_table['error_%'] = best_test_table['absdiff']/best_test_table['truth']*100
    best_test_table = best_test_table.sort_values(by=['error_%'], ascending=False).round({'absdiff':2,"error_%":2}) 

    grouped_test['absdiff'] = abs(grouped_test['prediction'] - grouped_test['truth'])
    grouped_test['error_%'] = abs(grouped_test['prediction'] - grouped_test['truth'])/grouped_test['truth']*100
    grouped_test = grouped_test.sort_values(by=['error_%'], ascending=False).round({'absdiff':2,"error_%":2}) 

    best_train_table['absdiff'] = abs(best_train_table['prediction'] - best_train_table['truth'])
    best_train_table['error_%'] =  best_train_table['absdiff']/best_train_table['truth']*100
    best_train_table = best_train_table.sort_values(by=['error_%'], ascending=False).round({'absdiff':2,"error_%":2}) 

    grouped_train['absdiff'] = abs(grouped_train['prediction'] - grouped_train['truth'])
    grouped_train['error_%'] = grouped_train['absdiff']/grouped_train['truth']*100
    grouped_train = grouped_train.sort_values(by=['error_%'], ascending=False).round({'absdiff':2,"error_%":2}) 

    # Calculate error by sector, sector no, weekday
#     groupBySector_test = best_test_table.groupby(['TO_SECTOR','SEAT_CLASS','Distance']).agg(
#         {,'error_%':"mean"}
#     ).sort_values(by=['error_%'], ascending=False).round({"error_%":2})
    f = lambda x: sum(x)
    f.__name__ = 'unique'

    groupBySector_test = best_test_table.groupby(['TO_SECTOR','SEAT_CLASS','Distance'],as_index=False).agg(
        {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupBySector_test['error1_%']=(groupBySector_test['absdiff']/groupBySector_test['truth'])*100
    groupBySector_test = groupBySector_test.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
    groupBySector_train = best_train_table.groupby(['TO_SECTOR','SEAT_CLASS','Distance'],as_index=False).agg(
        {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupBySector_train['error1_%']=(groupBySector_train['absdiff']/groupBySector_train['truth'])*100
    groupBySector_train = groupBySector_train.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
  

    groupBySectorNo_test = best_test_table.groupby(['SECTOR_NO','SEAT_CLASS'],as_index=False).agg(
        {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupBySectorNo_test['error1_%']=(groupBySectorNo_test['absdiff']/groupBySectorNo_test['truth'])*100
    groupBySectorNo_test = groupBySectorNo_test.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
    groupBySectorNo_train = best_train_table.groupby(['SECTOR_NO','SEAT_CLASS'],as_index=False).agg(
        {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupBySectorNo_train['error1_%']=(groupBySectorNo_train['absdiff']/groupBySectorNo_train['truth'])*100
    groupBySectorNo_train = groupBySectorNo_train.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
    
    groupByWeekday_test = best_test_table.groupby(['weekday','SEAT_CLASS'],as_index=False).agg(
        {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupByWeekday_test['error1_%']=(groupByWeekday_test['absdiff']/groupByWeekday_test['truth'])*100
    groupByWeekday_test = groupByWeekday_test.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
    groupByWeekday_train = best_train_table.groupby(['weekday','SEAT_CLASS'],as_index=False).agg(
      {'truth':"mean",'absdiff':"mean",'error_%':"mean"})
    groupByWeekday_train['error1_%']=(groupByWeekday_train['absdiff']/groupByWeekday_train['truth'])*100
    groupByWeekday_train = groupByWeekday_train.rename(columns={'error_%': 'error2_%'}).sort_values(
        by=['error1_%'], ascending=False).round({'truth':2,'absdiff':2,"error1_%":2,"error2_%":2})
    
    # Output to csv files
    best_test_table.to_csv(os.path.join(detail_directory,"PerFlightError_test.csv"))
    grouped_test.to_csv(os.path.join(detail_directory,"PerDayError_test.csv"))
    groupBySector_test.to_csv(os.path.join(detail_directory,"PerSectorError_test.csv"))
    groupBySectorNo_test.to_csv(os.path.join(detail_directory,"PerSectorNoError_test.csv"))
    groupByWeekday_test.to_csv(os.path.join(detail_directory,"PerWeekdayError_test.csv"))
    
    best_train_table.to_csv(os.path.join(detail_directory,"PerFlightError_train.csv"))
    grouped_train.to_csv(os.path.join(detail_directory,"PerDayError_train.csv"))
    groupBySector_train.to_csv(os.path.join(detail_directory,"PerSectorError_train.csv"))
    groupBySectorNo_train.to_csv(os.path.join(detail_directory,"PerSectorNoError_train.csv"))
    groupByWeekday_train.to_csv(os.path.join(detail_directory,"PerWeekdayError_train.csv"))
        
    # Calculate average over n runs
    try:
        mae = round(sum(test_flight_mae_list) / float(len(test_flight_mae_list)),5)
        sd = round(sum(test_flight_sd_list) / float(len(test_flight_sd_list)),5)
        mae_sumToDay = round(sum(test_day_mae_list) / float(len(test_day_mae_list)),5)
        sd_sumToDay = round(sum(test_day_sd_list) / float(len(test_day_sd_list)),5)
        #mae_sector_
    except ZeroDivisionError:
        print("[ERROR] MAE did not converge, failed to train model!")

    # Write results to txt file
    with open(txt_path, "a") as text_file:        
        text_file.write("Flight error%       : {}\n".format(mae*100))
        text_file.write("Flight sd          : {}\n".format(sd))
        text_file.write("-" * 50)
        text_file.write("\n")
        text_file.write("Day error%    : {}\n".format(mae_sumToDay*100))
        text_file.write("Day sd       : {}\n".format(sd_sumToDay))
        text_file.write("-" * 50)
        text_file.write("\n")
        text_file.write("sector1 error%    : {}\n".format(
            groupBySectorNo_test[(groupBySectorNo_test.SECTOR_NO == 1)]['error1_%'].values[0]
        ))
        text_file.write("sector2 error%    : {}\n".format(
            groupBySectorNo_test[(groupBySectorNo_test.SECTOR_NO == 2)]['error1_%'].values[0]
        ))
        text_file.write("-" * 50)
        text_file.write("\n")
        text_file.write("Mon error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 1)]['error1_%'].values[0]
        ))
        text_file.write("Tue error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 2)]['error1_%'].values[0]
        ))
        text_file.write("Wed error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 3)]['error1_%'].values[0]
        ))
        text_file.write("Thu error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 4)]['error1_%'].values[0]
        ))
        text_file.write("Fri error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 5)]['error1_%'].values[0]
        ))
        text_file.write("Sat error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 6)]['error1_%'].values[0]
        ))
        text_file.write("Sun error%    : {}\n".format(
            groupByWeekday_test[(groupByWeekday_test.weekday == 7)]['error1_%'].values[0]
        ))
    print("\n")
    print("Experiment    : {}\n".format(exp.getExpNo()))
    print("Flight error% : {}".format(mae*100))
    print("Flight sd     : {}".format(sd))
    print("Day error%    : {}".format(mae_sumToDay*100))
    print("Day sd        : {}".format(sd_sumToDay))
    print("\n-----EXPERIMENT DONE-----\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




