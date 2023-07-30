import pandas as pd
data_train=pd.read_csv('D:\desktop\接近（几乎）任何机器学习问题\project1\input\mnist_train.csv')
data_train['kfold']=-1
data_train=data_train.sample(frac=1).reset_index(drop=True)
y=data_train.label.values
from sklearn.model_selection import KFold
skf=KFold(n_splits=5)
for f,(t_,v_) in enumerate(skf.split(X=data_train,y=y)):
    data_train.loc[v_,'kfold']=f
data_train.to_csv('D:\desktop\接近（几乎）任何机器学习问题\project1\input\mnist_train_folds.csv')



