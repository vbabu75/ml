import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
def generate_store_sales_data(nStores,nDays,filename):
    '''
    Generates store sales data that is dependent on store, month and 
    weekday
    '''
    wday_weights = np.zeros((nStores,7))
    epoch_weights = np.zeros((nStores,120)) #Upto 10 years of data
    for store in range(nStores):
        wday_weights[store,:] = np.random.randint(1,10,7)
        epoch_weights[store,:] = np.random.randint(1,5,120)
    rec_array = np.zeros((nStores,nDays))
    start_date = (datetime.today().date())-timedelta(days=nDays)
    index=0
    store_ids=[]
    dates=[]
    sales=[]
    for store_id in range(nStores):
        curr_date = start_date
        for nDay in range(nDays):
            idx_wday = int(curr_date.weekday())
            idx_epoch = int(((curr_date-start_date).days)/30)
            curr_sales = ((100*epoch_weights[store_id,idx_epoch])+
                          (10*wday_weights[store_id,idx_wday])+
                          np.random.random()*2).round(2)
            store_ids.append("Store_"+str(store_id+1))
            sales.append(curr_sales)
            dates.append(curr_date)
            curr_date += timedelta(days=1)
    df = pd.DataFrame()
    df['store_id']=store_ids
    df['date']=dates
    df['sales']=sales
    df.to_csv(filename,index=False)
    print("Done generating store sales data")
    
if __name__=='__main__':
    generate_store_sales_data(200,1000,'store_sales.csv')
