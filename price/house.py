#Importing Libraries
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle

def stringToInt(string):
    integer = 0
    #try seeing if string value given is already a number if so the output would be
    try:
        #means string value given is already a int
        integer = int(string)
    except:
        string = string.lower()
        for i in string:
            integer += ord(i)
    return integer



#Read the file
df1 = pd.read_csv(r"C:\Users\preet\Downloads\Datasets (1)\final data set\Bengaluru_House_Data.csv")

#shape of data
df1.shape

#Information Of Data
df1.info

#DataTypes Printing
print(df1.dtypes)

df1.groupby('area_type')['area_type'].agg('count')

#Creating new datafame df2
#drop the unwanted features 
df2 = df1.drop(['area_type','availability','colony','balcony'], axis = 'columns')
df2.head()

#finding is any NULL value
df2.isnull().sum()

#Creating New Dataframe df3
df3 = df2.dropna()
df3.isnull().sum()

#finding the unique values in Feature "SIZE"
df3['size'].unique()

#Apply lambda function
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

#Head of the datdframe
df3.head()

#Finding unique value in Feature "BHK"
df3['bhk'].unique()

df3[df3.bhk>20]

#finding the unique value in Feature "total_sqft"
df3.total_sqft.unique()

#checking the variations in Total_sqft
def is_float(x):
        try:
            float(x)
        except:
            return False
        return True
       
df3[~df3['total_sqft'].apply(is_float)].head(15)

#converting range into single number
def converts_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return(float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

converts_sqft_to_num('1160')

converts_sqft_to_num('1195 - 1440')

converts_sqft_to_num('4125Perch')

#create new dataframe df4
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(converts_sqft_to_num)
df4.head(35)


#find particular location in feature "total_sqft"
df4.loc[672]

#create new dataframe df5
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()

#finding unique values in feature "LOCATION"
df5.location.unique()

#length of the feature "LOCATION"
len(df5.location.unique())

#Reducing the Diamensionality Curse
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


len(location_stats[location_stats<=10])

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

len(df5.location.unique())

df5.location = df5.location.apply(lambda x: 'Other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


#make a new category "OTHER"
df5.head(20)

#Now We find the Outliers in our dataset
df5[df5.total_sqft/df5.bhk<300].head()

#checking number of rows in dataset 
df5.shape

#now see we remove some outliers
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape

#description of the price_per_sqft(Statical values)
df6.price_per_sqft.describe()

#now we check outliers in feature "price_per_sqft"
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduce_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduce_df], ignore_index=True)
    return df_out

#Create a new dataframe df7
df7 = remove_pps_outliers(df6)

#checking the shape of dataframe
df7.shape


#Now we want to remove outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
                
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = 'index')

#creating new dataframe df8
df8 = remove_bhk_outliers(df7)
df8.shape


#Let's Explore bathroom feature
df8.bathrooms.unique()


df8[df8.bathrooms>10]

df8[df8.bathrooms>df8.bhk+2]

df8.shape

#creating a new dataframe df9
#remove bath outliers also
df9 = df8[df8.bathrooms<df8.bhk+2]
df9.shape

#creating a new dataset df10
#drop size and price_per_sqft features
df10 = df9.drop(['size','price_per_sqft'],axis = 'columns')
df10.head()


#creating dummies in location
dummies = pd.get_dummies(df10.location)
dummies.head(10)


#append dummies values in main dataframe
#creating new dataframe df11
df11 = pd.concat([df10,dummies.drop('Other',axis = 'columns')],axis = 'columns')
df11.head()



#creating a new dataframe df12
df12 = df11.drop('location',axis = 'columns')
df12.head()

#remove independent variable fron dataset
X = df12.drop('price', axis = 'columns')
X.head(2)


#making indepenent variable
y = df12.price
y.head()

#train_test_split method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)

#making model using Linear_Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.score(X_test,y_test)

X.columns

#Predicting Model Value
def predict_price(location,sqft,bath,bhk):
	loc_index = np.where(X.columns==location)[0][0]

	x = np.zeros(len(X.columns))
	x[0] = sqft
	x[1] = bath
	x[2] = bhk
	if loc_index >=0:
	    x[loc_index] = 1


    
	return lm.predict([x])[0]



	


	






























































































































































































































































	    
		













