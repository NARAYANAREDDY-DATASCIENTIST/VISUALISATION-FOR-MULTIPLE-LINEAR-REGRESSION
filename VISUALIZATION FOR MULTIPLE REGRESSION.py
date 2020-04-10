# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:37:27 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""
# VISUALIZATION FOR MULTIPLE REGRESSION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=[[150,178],[152,179],[179,200],[164,190],[180,220],[164,200],[159,189],[163,210],[181,270]]
Y=[0.75,0.89,1.02,0.34,0.54,0.67,0.94,0.87,0.69]

# PREPARE THE DATASET
df=pd.DataFrame(X,columns=['Price','AdSpends'])
df['Sales']=pd.Series(Y)

# BUILD THE MULTIPLE LINEAR REGRESSION

import statsmodels.formula.api as smf
model=smf.ols(formula='Sales ~ Price + AdSpends',data=df)
results_formula=model.fit()
results_formula.params

# VISUALIZING THE MULTIPLE LINEAR REGRESSION

x_surf,y_surf=np.meshgrid(np.linspace(df.Price.min(),df.Price.max(),100),np.linspace(df.AdSpends.min(),df.AdSpends.max(),100))
onlyX=pd.DataFrame({'Price':x_surf.ravel(), 'AdSpends':y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)

# COVERT THE PREDICTED REULTS IN A ARRAY
fittedY=np.array(fittedY)

# VISUALIZE THE MULTIPLE LINEAR REGRESSION
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(df['Price'],df['AdSpends'],df['Sales'],c='red',marker='o',alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape),color='None',alpha=0.3)
ax.set_xlabel=('Price')
ax.set_ylabel=('AdSpends')
ax.set_zlabel=('Sales')
plt.show()
