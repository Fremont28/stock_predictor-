#creating classes for coins
import pandas as pd 
import numpy as np 
import scipy 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from matplotlib.backends.backend_pdf import PdfPages
import sys 

sys.stdout = open("c:\\coins_risk1.txt", "w")
print ("test sys.stdout")

#define scaler (globally)
sc=StandardScaler()

##read-in the coin data
coins=pd.read_csv("coin_supply1.csv")
#coins1=coins 
#calculate coin market cap 
coins1['coin_market_cap']=coins1['available_supply']*coins1['price_btc']
#categorical to numeric 
coins1.proof_type = pd.Categorical(coins1.proof_type)
coins1.proof_type=coins1.proof_type.cat.codes
coins1.algorithm = pd.Categorical(coins1.proof_type)
coins1.algorithm=coins1.algorithm.cat.codes
#how long has coin been in service? (service nj gas stations?)
coins1.start_date = pd.Categorical(coins1.start_date)
coins1.start_date=coins1.start_date.cat.codes
#social sentiment
coins1.social_sentiment = pd.Categorical(coins1.social_sentiment)
coins1.social_sentiment=coins1.social_sentiment.cat.codes
#category (currency vs. payments?)
coins1.category = pd.Categorical(coins1.category)
coins1.category=coins1.category.cat.codes
#github status
coins1.github_status = pd.Categorical(coins1.github_status)
coins1.github_status=coins1.github_status.cat.codes
#white paper
coins1.WhitePaperAvailability = pd.Categorical(coins1.WhitePaperAvailability)
coins1.WhitePaperAvailability=coins1.WhitePaperAvailability.cat.codes
#roadmap
coins1.RoadmapAvailability = pd.Categorical(coins1.RoadmapAvailability)
coins1.RoadmapAvailability=coins1.RoadmapAvailability.cat.codes
#product status
coins1.ProductStatus = pd.Categorical(coins1.ProductStatus)
coins1.ProductStatus=coins1.ProductStatus.cat.codes
#token function (nj ham vs. self-service mechanics?)
coins1.TokenFunction = pd.Categorical(coins1.TokenFunction)
coins1.TokenFunction=coins1.TokenFunction.cat.codes
#partnership
coins1.PartnershipTraction = pd.Categorical(coins1.PartnershipTraction)
coins1.PartnershipTraction=coins1.PartnershipTraction.cat.codes
#user case assessment 
coins1.UseCaseAssessment = pd.Categorical(coins1.UseCaseAssessment)
coins1.UseCaseAssessment=coins1.UseCaseAssessment.cat.codes

#7-day absolute price change (in coin)
coins1['percent_30_day_change']=coins1['PriceMovement_Price_30day']/coins1['price_usd']
#covvert to absolute value
coins1['percent_30_day_change']=coins1['percent_30_day_change'].abs()

#percent price change (assign a threshold)
coins1['percent_30_day_change_bin']=coins1.percent_30_day_change.apply(lambda x: 1 if x>61.5 else 0)
coins1['percent_30_day_change']=coins1['percent_30_day_change'].fillna(0)

coins2=coins1[["symbol","available_supply","price_usd","price_btc","volume_usd_24h","market_cap_usd","coin_market_cap","total_supply","max_supply","algorithm",
"start_date","percent_30_day_change_bin","PriceMovement_Price_90day","PriceMovement_Price_30day"]]
######PRICE CHANGES 90 and 30 DAYS***********************
coins2['PriceMovement_Price_90day']=coins2['PriceMovement_Price_90day'].fillna(0)
coins2['PriceMovement_Price_30day']=coins2['PriceMovement_Price_30day'].fillna(0)

#90-day price movement 
coins2['price_change_90']=coins2['PriceMovement_Price_90day']/coins2['price_usd']
coins2['price_change_90']=coins2['price_change_90'].fillna(0)
coins2['price_change_90']=abs(coins2['price_change_90']) #absolute value 
#30-day price movement
coins2['price_change_30']=coins2['PriceMovement_Price_30day']/coins2['price_usd']
coins2['price_change_30']=coins2['price_change_30'].fillna(0)
coins2['price_change_30']=abs(coins2['price_change_30']) #absolute value 

#threshold values (1-low risk to 5-high risk)*********************************
#a.threshold levels 90 days price movement 
def threshold_90(x):
    if x==0:
        return 1
    if x>0 and x<25:
        return 2
    if x>=25 and x<100:
        return 3
    if x>=100 and x<500:
        return 4
    else:
        return 5

#90-day volatility score 
coins2['90_day_vola_score']=coins2['price_change_90'].apply(threshold_90)

#b. threshold levels 30 days price movement 
def threshold_30(x):
    if x==0:
        return 1
    if x>0 and x<10:
        return 2
    if x>=10 and x<50:
        return 3
    if x>=50 and x<100:
        return 4
    else:
        return 5 

#30-day volatility score 
coins2['30_day_vola_score']=coins2['price_change_30'].apply(threshold_30)

#average 90 and 30 day volatility
coins2['avg_volatility']=(coins2['90_day_vola_score']+coins2['30_day_vola_score'])/2
coins2['avg_volatility']=round(coins2['avg_volatility'],0)

coins2=coins2.drop('PriceMovement_Price_90day',1)
coins2=coins2.drop('PriceMovement_Price_30day',1)
coins2=coins2.drop('start_date',1)

########PART 2 read-in the time series data **********************************************
coin_hist=pd.read_csv("coin_daily_histories.csv")

#group by max price and min price for each coin
max_price=coin_hist.groupby('symbol')['value'].max()
max_price.reset_index(level=0)
max_price=pd.DataFrame(max_price)
max_price.reset_index(level=0,inplace=True)
max_price.columns=['symbol','max_value']
min_price=coin_hist.groupby('symbol')['value'].min()
min_price.reset_index(level=0)
min_price=pd.DataFrame(min_price)
min_price.reset_index(level=0,inplace=True)
min_price.columns=['symbol','min_value']
avg_price=coin_hist.groupby('symbol')['value'].mean() 
avg_price.reset_index(level=0)
avg_price=pd.DataFrame(avg_price)
avg_price.reset_index(level=0,inplace=True)
avg_price.columns=['symbol','avg_value']

#standard deviation of price (usd) by coin
coin_ts=coin_hist[['symbol','value']]
coin_ts=coin_ts.set_index('symbol')

w=10 #10-day moving average price 
roller=coin_hist.rolling(w)
vol_list=roller.std(ddof=0)
vol_list1=vol_list.reset_index()

sd_price=vol_list1.groupby('symbol')['value'].mean()
sd_price1=sd_price.reset_index()
sd_price1.columns=['symbol','std_dev']

coin_value=coin_hist[['symbol','value']]
coin_value=coin_value.groupby('symbol')['value'].mean()
coin_value.reset_index(level=0) 
coin_value=pd.DataFrame(coin_value)
coin_value.reset_index(level=0,inplace=True)

#merge dataframes (max and min prices for each coin)
final_value=pd.merge(max_price,min_price,on='symbol') #time series 
final_value1=pd.merge(final_value,avg_price,on='symbol')

#merge df
final_value2=pd.merge(sd_price1,final_value1,on="symbol")

#standard deviation change
final_value2['percent_sd']=final_value2['std_dev']/final_value2['avg_value']
#percent max-min/avg
final_value2['value_volatility_score']=(final_value2['max_value']-final_value2['min_value'])/(final_value2['avg_value'])

#merge coins2 and final_value1 dataframe?*************************
final=pd.merge(final_value2,coins2,on="symbol") #unscaled data*************************************
#split into scale and unscaled data (re-merge)
final=final.replace([np.inf,-np.inf],np.nan)
final=final.fillna(0)

#drop columns 90 and 30-day price movement (categorical values)
final=final.drop('90_day_vola_score',1)
final=final.drop('30_day_vola_score',1)

names=final["symbol"] #coin symbols
avg_volatility=final["avg_volatility"]
#scale_data=pd.concat([scale_data,avg_volatility],axis=1)
final1=pd.concat([names,final],axis=1)

coins3=coins1
#average volatility score (90 and 30 day price movements for each coin)
coins3['PriceMovement_Price_90day']=coins3['PriceMovement_Price_90day'].fillna(0)
coins3['PriceMovement_Price_30day']=coins3['PriceMovement_Price_30day'].fillna(0)
#90-day price movement 
coins3['price_change_90']=coins3['PriceMovement_Price_90day']/coins3['price_usd']
coins3['price_change_90']=coins3['price_change_90'].fillna(0)
coins3['price_change_90']=abs(coins3['price_change_90']) #absolute value 
#30-day price movement
coins3['price_change_30']=coins3['PriceMovement_Price_30day']/coins3['price_usd']
coins3['price_change_30']=coins3['price_change_30'].fillna(0)
coins3['price_change_30']=abs(coins3['price_change_30']) #absolute value

#threshold values (1-low risk to 5-high risk)********** final1
#a.threshold levels 90 days price movement 
def threshold_90(x):
    if x==0:
        return 1
    if x>0 and x<25:
        return 2
    if x>=25 and x<100:
        return 3
    if x>=100 and x<500:
        return 4
    else:
        return 5

#90-day volatility score 
coins3['90_day_vola_score']=coins3['price_change_90'].apply(threshold_90)

#b. threshold levels 30 days price movement 
def threshold_30(x):
    if x==0:
        return 1
    if x>0 and x<10:
        return 2
    if x>=10 and x<50:
            return 3
    if x>=50 and x<100:
        return 4
    else:
        return 5 

#30-day volatility score 
coins3['30_day_vola_score']=coins3['price_change_30'].apply(threshold_30)

#average 90 and 30 day volatility scores 
coins3['avg_volatility']=(coins3['90_day_vola_score']+coins3['30_day_vola_score'])/2
coins3['avg_volatility']=round(coins3['avg_volatility'],0)
coins3=coins3.drop('PriceMovement_Price_90day',1)
coins3=coins3.drop('PriceMovement_Price_30day',1)

#merge dataframes 
final_test=pd.merge(coins3,final_value2,on="symbol")
#read in new dataframe (risk scores)
metrics=pd.read_csv("coin_metrics.csv")
metrics1=metrics.iloc[1:metrics.shape[0]]
metrics2=metrics1.iloc[:,0:24]
metrics2=metrics1[["Unnamed: 0","Unnamed: 2","Unnamed: 3","Unnamed: 17","Unnamed: 21"]] #id, name,symbol,sharpe ratio, mdd
metrics2.columns=['id','name','symbol','sharpe_ratio','mdd']
metrics2=metrics2.fillna(0)
#merge dataframes 
final_test1=pd.merge(final_test,metrics2,on="symbol")


class Coins_Analysis():
    def __init__(self):
        pass

    def average_risk(self):
        #split  data into market risk, liquidity risk, and issuer risk ********** 
        #i. market risk #percent_sd, sharpe_ratio
        market=final_test1[["symbol","mdd","percent_30_day_change_bin","avg_volatility"]]
        market1=market.iloc[:,1:market.shape[1]]
        market1=market1.astype(float)
        market1=market1.abs()
        symbol=market['symbol']
        symbol=pd.DataFrame(symbol,columns=['symbol'])
        market_final=pd.concat([symbol,market1],axis=1)

        X=market_final.iloc[:,0:market.shape[1]-1]
        y=market_final[["symbol","avg_volatility"]]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=101)

        X_train1=X_train.iloc[:,1:X_train.shape[1]-1]
        X_train1=X_train1.fillna(0)
        X_test1=X_test.iloc[:,1:X_test.shape[1]-1]
        y_train1=y_train["avg_volatility"] 
        y_train1=pd.DataFrame(y_train1)
        y_test1=y_test["avg_volatility"]
        y_test1=pd.DataFrame(y_test1)

        #scale dataset **************
        sc=StandardScaler()
        X_train_scale=sc.fit_transform(X_train1)
        X_train_scale=pd.DataFrame(X_train_scale)
        X_test_scale=sc.fit_transform(X_test1)
        X_test_scale=pd.DataFrame(X_test_scale)
         
        #classification random forest 
        model=RandomForestClassifier(n_estimators=500) 
        model.fit(X_train_scale,y_train1)
        predict=model.predict(X_test_scale) #classify 
        predict1=pd.DataFrame(predict,columns=['risk_score1'])

        #combine predict1 and X_test
        X_test=X_test.reset_index(level=0,inplace=False)
        combine_predictions=pd.concat([predict1,X_test],axis=1)
        combine_predictions1=pd.merge(combine_predictions,y_test,on="symbol")
        combine_predictions1=combine_predictions1[["symbol","risk_score1"]]

        #ii. liquidity risk ******************
        liquid=final_test1[["symbol","market_cap_usd","volume_usd_24h","available_supply","total_supply","avg_volatility"]]
        X=liquid.iloc[:,0:liquid.shape[1]-1]
        y=liquid[["symbol","avg_volatility"]]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=101)

        X_train1=X_train.iloc[:,1:X_train.shape[1]-1]
        X_train1=X_train1.fillna(0)
        X_test1=X_test.iloc[:,1:X_test.shape[1]-1]
        y_train1=y_train["avg_volatility"] 
        y_train1=pd.DataFrame(y_train1)
        y_test1=y_test["avg_volatility"]
        y_test1=pd.DataFrame(y_test1)

        #scale liquidity dataset 
        sc=StandardScaler()
        X_train_scale=sc.fit_transform(X_train1)
        X_train_scale=pd.DataFrame(X_train_scale)
        X_test_scale=sc.fit_transform(X_test1)
        X_test_scale=pd.DataFrame(X_test_scale)

        #classification random forest 
        model=RandomForestClassifier(n_estimators=500,max_features=3) 
        model.fit(X_train_scale,y_train1)
        predict2=model.predict(X_test_scale) #classify 
        predict3=pd.DataFrame(predict,columns=['risk_score2'])

        #combine predict and X_test
        X_test=X_test.reset_index(level=0,inplace=False)
        combine_predictions2=pd.concat([predict3,X_test],axis=1)
        combine_predictions3=pd.merge(combine_predictions2,y_test,on="symbol")
        combine_predictions4=combine_predictions3[["symbol","risk_score2"]]

        #iii. issuer risk *************************
        issuer=final_test1[["symbol","algorithm","social_sentiment","category","github_status","WhitePaperAvailability","RoadmapAvailability","ProductStatus","TokenFunction","PartnershipTraction","UseCaseAssessment","avg_volatility"]]

        X=issuer.iloc[:,0:issuer.shape[1]-1]
        y=issuer[["symbol","avg_volatility"]]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=101)

        X_train1=X_train.iloc[:,1:X_train.shape[1]-1]
        X_train1=X_train1.fillna(0)
        X_test1=X_test.iloc[:,1:X_test.shape[1]-1]
        y_train1=y_train["avg_volatility"] 
        y_train1=pd.DataFrame(y_train1)
        y_test1=y_test["avg_volatility"]
        y_test1=pd.DataFrame(y_test1)

        X_train_scale=sc.fit_transform(X_train1)
        X_train_scale=pd.DataFrame(X_train_scale)
        X_test_scale=sc.fit_transform(X_test1)
        X_test_scale=pd.DataFrame(X_test_scale)

        model=RandomForestClassifier(n_estimators=500,max_features=5) 
        model.fit(X_train_scale,y_train1)
        predict6=model.predict(X_test_scale) #classify 
        predict7=pd.DataFrame(predict6,columns=['risk_score3'])

        #combine predict3 and X_test
        X_test=X_test.reset_index()
        combine_predictions5=pd.concat([predict7,X_test],axis=1)
        combine_predictions6=pd.merge(combine_predictions5,y_test,on="symbol")
        combine_predictions7=combine_predictions6[["symbol","risk_score3"]]

        ####all risk score predictions/merge and average 
        merge_preds=pd.merge(combine_predictions1,combine_predictions4,on="symbol")
        merge_preds1=pd.merge(merge_preds,combine_predictions7,on="symbol")
        merge_preds1['risk_score']=(merge_preds1.risk_score1+merge_preds1.risk_score2+merge_preds1.risk_score3)/3
        return merge_preds1 


if __name__ =='__main__':
    c=Coins_Analysis()
    risk_scores=c.average_risk()