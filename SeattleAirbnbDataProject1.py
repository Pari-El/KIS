#!/usr/bin/env python
# coding: utf-8

# In[504]:


#FIrst Look at the DATA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from math import sin, cos, sqrt, atan2, radians
get_ipython().run_line_magic('matplotlib', 'inline')

# 1) Does the proximity to city center mean higher price?
# 2) I tend to find it easy to book a place with high amount of reviews, distance to city centre and price.lets investigate 
    # this reality
# 3) where is the best location, pricing and property type?


df_listings = pd.read_csv('./Desktop/KIS/seattle/listings.csv')

# -- CHECK COLUMNS TYPEs & MISSING DATA -- 
def df_cat_and_qnt_details(df):
    
    '''
    INPUT - df - pandas dataframe which is read csv
    OUTPUT - return the lists of column names of both full and non-full categorical and quantitative columns 
        [0] df_cat_all_Values_ColName,
        [1] df_cat_missing_Values_ColName, 
        [2] df_qnt_all_Values_ColName, 
        [3] df_qnt_missing_Values_ColName    
    '''  
    print("rows: ", df.shape[0], "Columns: ", df.shape[1])
    
    print("- Identify categorical columns in the DFs -")
    df_cat = df.select_dtypes(include=['object'])
    print("categorical columns: ", df_cat.shape[1]) 
    
    print("- categorical columns with no missing values -")
    df_cat_all_Values = np.sum(np.sum(df_cat.isnull())/df_cat.shape[0] == 0)
    print("Categorical Columns with all data: ", df_cat_all_Values)
    
    #categorical columns Names with no missing values
    df_cat_all_Values_ColName = df_cat.columns[np.sum(df_cat.isnull())==0].tolist()
    #categorical columns Names with missing values
    df_cat_missing_Values_ColName = df_cat.columns[np.sum(df_cat.isnull())!=0].tolist()
    
    print("- Identify Quantitative columns in the DFs -")
    df_qnt = df.select_dtypes(include=['int64', 'float64'])
    print("Quantitative Columns: ", df_qnt.shape[1])
    
    print("- Quantitative columns Names with no missing values -")
    df_qnt_all_Values = np.sum(np.sum(df_qnt.isnull())/df_qnt.shape[0] == 0)
    print("Quantitative Columns with all data:", df_qnt_all_Values)
    print()
    
    #Quantitative columns Names with missing values
    df_qnt_all_Values_ColName = df_qnt.columns[np.sum(df_qnt.isnull())==0].tolist()
    #Quantitative columns Names with missing values
    df_qnt_missing_Values_ColName = df_qnt.columns[np.sum(df_qnt.isnull())!=0].tolist()
    
    #return the columns names of both full and non-full categorical and quantitative columns
    return df_cat_all_Values_ColName, df_cat_missing_Values_ColName, df_qnt_all_Values_ColName, df_qnt_missing_Values_ColName


# In[505]:


#print the list of column names which are Categorical with all values and quantitative with all values in the 'listings' and 'review' data
print("listings")
df_listings_dets = df_cat_and_qnt_details(df_listings)


# In[623]:


#[0] df_cat_all_Values_ColName, [1] df_cat_missing_Values_ColName, [2] df_qnt_all_Values_ColName, [3] df_qnt_missing_Values_ColName 
df_listings_dets


# In[507]:


df_listings.head()


# In[508]:


#Column 'Price' is considered a Categorical Column, but it is expected to be a quantitative as an amount and is a column of interest
print(pd.DataFrame(df_listings, columns=['price']))


# In[509]:


#want to convert the string/object format of the price into a float which can be made for quantitative purpose
df_listings['price'] = [price.strip('$') for price in df_listings.price]
df_listings['price'] = [price.replace(',', '') for price in df_listings.price]
df_listings['price'] = df_listings.price.astype(float)
print(pd.DataFrame(df_listings, columns=['price']), df_listings['price'].dtypes)


# In[510]:


#Now that the 'price' column is quantitative, we can have a look at the columns we're interested in
df_listings.drop(df_listings.columns.difference(['review_scores_rating', 'number_of_reviews', 
                                                 'neighbourhood', 'property_type','room_type', 'bed_type','latitude','longitude','price']), axis=1, inplace=True)


# In[577]:


#here, we will add an additional Column which we can use the below Lat/Lon distance between 
#the city center and the neighbourhoods to calculate the distance

#Seattle City Center: 47.6103 lat and 122.3341 long

# approximate radius of earth in km
R = 6373.0
seattle_CC_lat = radians(47.6103)
seattle_CC_lon = radians(-122.3341)
dist_ary = np.array([])
for index, lat_long in df_listings.iterrows(): 
    neighbourhood_lat = radians(lat_long['latitude'])
    neighbourhood_lon = radians(lat_long['longitude'])
    dlon = neighbourhood_lon - seattle_CC_lon #differece in longitude
    dlat = neighbourhood_lat - seattle_CC_lat #differece in latitude
    a = sin(dlat / 2)**2 + cos(seattle_CC_lat) * cos(neighbourhood_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    dist_ary = np.append(dist_ary, round(R * c, 2)) #appending calculated distance in km into an array

#append the distance array to the listings df. first transpose from row to column
print(dist_ary.shape)
dist_ary_a = np.reshape(dist_ary, (-1, 1))
print(dist_ary_a.shape)
print(df_listings.shape)
pd.concat([df_listings, pd.DataFrame(dist_ary_a.transpose())], axis=1)
df_listings_withDist  = pd.concat([df_listings, pd.DataFrame(dist_ary_a)], axis=1)

df_listings_withDist = df_listings_withDist.rename(columns = {0: "dist"}) 
df_listings_withDist.head()


# In[578]:


#Plot price vs distance for Question 1
df_listings_withDist.plot(x='dist', y='price', style='o');


# In[579]:


print("We can certainly see some visual corrolation between price and distance as the higher prices which are on the vertical are scattered closer to the left - closer to the city centre = 0. As a side task, we also want to see any corrlation between number of reviews and distance to city center, which i hypothesise should have a corrolation.")
print("number_of_reviews should be a quantitative full column mean:{}"
      .format(df_listings_withDist['number_of_reviews'].isnull().mean()))
    
df_listings_withDist.plot(x='dist', y='number_of_reviews', style='o');


# In[597]:


print("the above plot is very scattered to many any conclusions.")
# our 2nd QUESION will be to have an insight into the below
# MachineLearning will be used to see if there is any corrolation hidden within the variables to address why there is reviewing 
#traffic to come properties but not the others some categorical variables like property type, neighbourhood
# against some quantitative variables like distance to city centre, reviews and price.
sns.heatmap(df_listings_withDist.corr(), annot=True, fmt='.2f');


# In[598]:


#the heatmap indicates a general lack of corrolation but we will confirm with ML

#for the X and response Y vectors for ML. Here, i want the response vector to be 'number of reviews'
#First ensure all X vetors are the same without nan vales

print("df_listings_withDist")
df_listings_withDist_dets = df_cat_and_qnt_details(df_listings_withDist)
#[0] df_cat_all_Values_ColName, [1] df_cat_missing_Values_ColName, 
#[2] df_qnt_all_Values_ColName, [3] df_qnt_missing_Values_ColName 
df_listings_withDist_dets 


# In[599]:


#from the above, we can see that the categorial 
#['neighbourhood', 'property_type'] and quantitative ['review_scores_rating'] are missing some data
df_listings_withDist.isnull().sum(axis = 0)


# In[600]:


print("looking at the major missing data columns, we have neighbourhood and review_Scores_Raing. both missing over 5%")
print("we're going to Drop neighbourhoor and property because of their categorical nature which cannot be qantified")

#drop
df_listings_new = df_listings_withDist.drop(['property_type', 'neighbourhood'], axis=1)


# In[605]:


print("impute the mean of review_scores_rating into he column instead of dropping it. heatmap shows some importance to this col.")
df_listings_Full = df_listings_new.fillna(df_listings_new.review_scores_rating.mean()) #Fill all missing values with the mean of the column.
df_listings_Full.isnull().sum(axis = 0)


# In[609]:


#now that all the X vectors have values, we can compute the predition
X = df_listings_Full[['dist' ,'price', 'review_scores_rating']]
y = df_listings_Full['number_of_reviews']

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Instantiate
lm_model = LinearRegression(normalize=True)
#Fit
lm_model.fit(X_train, y_train)
        
#Predict and score the model
y_test_preds = lm_model.predict(X_test) 

#Rsquared and y_test
rsquared_score = r2_score(y_test, y_test_preds)
length_y_test = len(y_test)

"The r-squared score for your model was {} on {} values.".format(rsquared_score, length_y_test)


# In[586]:


#check the average price per neighbourhood based on the property and room type
pd.DataFrame(df_listings.groupby(['neighbourhood', 'property_type', 'room_type']).mean())['price']


# In[627]:


pd.DataFrame(df_listings_withDist.groupby(['property_type']).mean())['dist'].plot(kind="bar");

