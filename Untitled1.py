#!/usr/bin/env python
# coding: utf-8

# ### things to to : find out details of causal model
# 
# ### get rid of warnings
# 
# #### 원문코드는 띄어쓰기 틀린게 많음. PEP8을 참고하자. 기본중의 기본
# 
# 
# ### http://pywhy.org
# ### dataset is available from https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md
# 
# ### page from https://github.com/py-why/dowhy/blob/main/docs/source/example_notebooks/DoWhy-The%20Causal%20Story%20Behind%20Hotel%20Booking%20Cancellations.ipynb

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dowhy

# ... 지움
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[4]:


dataset = pd.read_csv('hotel_bookings.csv')
dataset.head()


# In[3]:


dataset.columns
len(dataset.columns)


# ## Feature Engineering
# 새로운 column을 추가하고 불필요한 column을 제거한다.
# 
# - Total Stay = stays_in_weekend_nights + stays_in_week_nights
# - Guests = adults + children + babies
# - Different_room_assigned = 1 if reserved_room_type & assigned_room_type are different, 0 otherwise.

# In[5]:


# Total stay in nights
dataset['total_stay'] = dataset['stays_in_week_nights'] + dataset['stays_in_weekend_nights']

# Total number of guests
dataset['guests'] = dataset['adults'] + dataset['children'] + dataset['babies']

# Creating the different_room_assigned feature
dataset['different_room_assigned'] = 0
slice_indices =dataset['reserved_room_type'] != dataset['assigned_room_type']
dataset.loc[slice_indices,'different_room_assigned'] = 1

# Deleting older features
dataset = dataset.drop(['stays_in_week_nights', 'stays_in_weekend_nights', 'adults','children','babies'
                        , 'reserved_room_type', 'assigned_room_type'], axis=1)
len(dataset.columns)


# In[6]:


dataset.isnull().sum() # Country, Agent, Company contain 488, 16340, 112593 missing entries 
dataset = dataset.drop(['agent','company'], axis=1)
# Replacing missing countries with most freqently occuring countries ===> not really good choice
dataset['country'] = dataset['country'].fillna(dataset['country'].mode()[0])


# ### 위의 코드의 경우 개선할 여지가 있음

# In[7]:


dataset = dataset.drop(
    ['reservation_status', 'reservation_status_date', 'arrival_date_day_of_month',
     'arrival_date_year', 'distribution_channel'],
    axis=1)


# In[8]:


# Replacing 1 by True and 0 by False for the experiment and outcome variables
dataset['different_room_assigned'] = dataset['different_room_assigned'].replace(1, True)
dataset['different_room_assigned'] = dataset['different_room_assigned'].replace(0, False)
dataset['is_canceled'] = dataset['is_canceled'].replace(1, True)
dataset['is_canceled'] = dataset['is_canceled'].replace(0, False)
dataset.dropna(inplace=True)
print(dataset.columns)
dataset.iloc[:, 5:20].head(100)


# In[9]:


dataset = dataset[dataset.deposit_type == "No Deposit"]
dataset.head()
dataset.groupby(['deposit_type', 'is_canceled']).count()


# In[10]:


dataset_copy = dataset.copy(deep=True)


# In[11]:


# Calculating Expected Counts
counts_sum = 0
for i in range(1, 10000):
        counts_i = 0
        rdf = dataset.sample(1000)
        counts_i = rdf[rdf["is_canceled"] == rdf["different_room_assigned"]].shape[0]
        counts_sum += counts_i
counts_sum / 10000


# In[12]:


# 표본 평균의 평균을 구한다
# magic number를 피해야 함
# 코드의 의미를 생각해보자
counts_sum = 0
num_repeats = 10000
for i in range(1, num_repeats):
        counts_i = 0
        rdf = dataset.sample(1000)
        counts_i = rdf[rdf["is_canceled"] == rdf["different_room_assigned"]].shape[0]
        counts_sum += counts_i
counts_sum / num_repeats # 표본 평균의 평균


# In[13]:


# 틀린 그림 찾기
counts_sum = 0
num_repeats = 10000
for i in range(1, num_repeats):
        counts_i = 0
        rdf = dataset[dataset["booking_changes"] > 0].sample(1000)
        counts_i = rdf[rdf["is_canceled"] == rdf["different_room_assigned"]].shape[0]
        counts_sum += counts_i
counts_sum / num_repeats # 표본 평균의 평균


# ### There is definitely some change happening when the number of booking changes are non-zero. So it gives us a hint that Booking Changes may be affecting room cancellation. <br>But is Booking Changes the only confounding variable? What if there were some unobserved confounders, regarding which we have no information(feature) present in our dataset. Would we still be able to make the same claims as before?

# In[15]:


import pygraphviz
causal_graph = """digraph {
different_room_assigned[label="Different Room Assigned"];
is_canceled[label="Booking Cancelled"];
booking_changes[label="Booking Changes"];
previous_bookings_not_canceled[label="Previous Booking Retentions"];
days_in_waiting_list[label="Days in Waitlist"];
lead_time[label="Lead Time"];
market_segment[label="Market Segment"];
country[label="Country"];
U[label="Unobserved Confounders",observed="no"];
is_repeated_guest;
total_stay;
guests;
meal;
hotel;
U->{different_room_assigned,required_car_parking_spaces,guests,total_stay,total_of_special_requests};
market_segment -> lead_time;
lead_time->is_canceled; country -> lead_time;
different_room_assigned -> is_canceled;
country->meal;
lead_time -> days_in_waiting_list;
days_in_waiting_list ->{is_canceled,different_room_assigned};
previous_bookings_not_canceled -> is_canceled;
previous_bookings_not_canceled -> is_repeated_guest;
is_repeated_guest -> {different_room_assigned,is_canceled};
total_stay -> is_canceled;
guests -> is_canceled;
booking_changes -> different_room_assigned; booking_changes -> is_canceled; 
hotel -> {different_room_assigned,is_canceled};
required_car_parking_spaces -> is_canceled;
total_of_special_requests -> {booking_changes,is_canceled};
country->{hotel, required_car_parking_spaces,total_of_special_requests};
market_segment->{hotel, required_car_parking_spaces,total_of_special_requests};
}"""


# In[16]:


model= dowhy.CausalModel(data=dataset, 
                         graph=causal_graph.replace("\n", " "),
                         treatment="different_room_assigned",
                         outcome='is_canceled')
model.view_model()
from IPython.display import Image, display
display(Image(filename="causal_model.png"))


# In[1]:


# Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
help(model.identify_effect)


# In[18]:


estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.propensity_score_weighting",
                                 target_units="ate")
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
print(estimate)


# In[19]:


refute1_results = model.refute_estimate(identified_estimand, estimate,
                                      method_name="random_common_cause")
print(refute1_results)


# In[20]:


refute2_results = model.refute_estimate(identified_estimand, estimate,method_name="placebo_treatment_refuter")
print(refute2_results)


# In[21]:


refute3_results=model.refute_estimate(identified_estimand, estimate,
                                      method_name="data_subset_refuter")
print(refute3_results)


# In[ ]:




