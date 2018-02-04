import xgboost
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X,y = make_regression(n_samples=1000,n_features=10)
trainX,valX,trainy,valy = train_test_split(X,y,test_size=0.2)

dtrain = xgboost.DMatrix(data=trainX,label=trainy)
dval = xgboost.DMatrix(data=valX,label=valy)


#'eta':0.3,'gamma':0,'max_depth':6,
       
param={'objective':'reg:linear'}
num_round=10
#evals = [(dtrain,'train'),(dval,'val')]
bst = xgboost.train(param,dtrain,num_round)#,evals=evals)