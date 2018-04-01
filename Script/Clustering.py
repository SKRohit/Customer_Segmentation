# -*- coding: utf-8 -*-
"""
This is the code to cluster retail customers into eight different clusters by applying hierarchical clustering on neurons
of a 2D self organizing map (som) obtained after training the som on a multi-dimensional retail customer data.
"""

import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from pylab import bone, pcolor, colorbar, plot, show
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from minisom import MiniSom
import re

#reads data and performs data cleaning
product = pd.read_csv('cproducts.csv')
product['customerID'] = [re.sub('BBID_', '', product['customerID'][i]) for x in product['customerID']]    
product['customerID'] = pd.to_numeric(product['customerID'])
#drops few redundant features
product.drop(['store_description','product_description', 'promotion_description', 'State'],axis = 1, inplace = True)  
 #drops dob and pincode as not good features
product.dropna(subset = ['DOB', 'PinCode'], inplace = True)  
product.reset_index(drop = True, inplace = True)
product['Gender'].fillna('0', inplace = True)
product['promo_code'].fillna(0, inplace = True)
product['DOB'].replace('NANA', np.nan, inplace = True) 
product['promo_code'].replace('NONPROMO', np.nan, inplace = True)
product['promo_code'].replace('LOCALPROMO', '1', inplace = True)
product['promo_code'] = pd.to_numeric(product['promo_code'])
product['transactionDate'] = pd.to_datetime(pd.Series(product['transactionDate']), format = '%Y/%m/%d')
product['DOB'] = pd.to_datetime(pd.Series(product['DOB']), format = '%Y/%m/%d')

#calculates age and add as features
def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
product['Age'] = [calculate_age(x) for x in product['DOB']]
product.drop(['DOB'],axis = 1, inplace = True)
product['Week'] = [x.isocalendar()[1] for x in product['transactionDate']]
product['Month'] = [x.month for x in product['transactionDate']]
product.drop(['transactionDate'],axis = 1, inplace = True)
product.dropna(subset = ['Age'], inplace = True)

#cleaned data with added features
product.to_csv('editedproducts2.csv', index = False)

#encoding categorical variables and standardising the data
class_le = LabelEncoder()
product['discountUsed'] = class_le.fit_transform(product['discountUsed'].values) 
product['Gender'] = class_le.fit_transform(product['Gender'].values) 
product.ix[:,[2,11]] = class_le.fit_transform(product.ix[:,[2,11]])
sc = StandardScaler()
product_std = sc.fit_transform(product)

#trains a 15 x 15 self organising map on cleaned standardised data
som = MiniSom(x = 15, y = 15, input_len = 13, sigma = 1.0)
som.random_weights_init(product_std)
som.train_random(product_std, 10000)

#plots the som map
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
for i, x in enumerate(product_std):
   w = som.winner(x)
  plot(w[0] +0.5, w[1]+0.5, markers[1], markeredgecolor = 'g', markerfacecolor = 'None', markersize = 7, markeredgewidth = 1)
show()
[plot(w[0] +0.5, w[1]+0.5, markers[1], markeredgecolor = 'g', markerfacecolor = 'None', markersize = 7, markeredgewidth = 1) for x in product_std ]

#weights of each trained neuron
weights = []
[weights.append(som.weights[i][j]) for i in range(15) for j in range(15)]
weights = np.array(weights)

#performs hierarchical clustering on neuron weights and plots a dendogram
Z = linkage(weights, 'average')
c, coph_dists = cophenet(Z, pdist(weights))
plt.figure(figsize=(25, 10))
plt.title('SOM Weights Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
den = dendrogram(Z, leaf_rotation = 90, leaf_font_size = 8, get_leaves = True)
plt.show()

#plots fance dendograms
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

fancy_dendrogram(Z, truncate_mode = 'lastp', p = 12, leaf_rotation = 90, 
                leaf_font_size = 12, show_contracted = True, 
                annotate_above = 10, max_d = 1.2)


mappings = []
[mappings.append((i,j)) for i in range(15) for j in range(15)]
#mapping of every data point on som
maps = som.win_map(product_std)

#decided eight clusters from the dendogram plot
#truncating the dendogram at a distance of 1.2
cluster = [[] for i in range(8)]
for i,x in enumerate(den['leaves']):
    if i<=44:
        cluster[0].append(maps[(mappings[x])])
    elif i>44 and i<=72:
        cluster[1].append(maps[(mappings[x])])
    elif i>72 and i<=100:
        cluster[2].append(maps[(mappings[x])])
    elif i>100 and i<=144:
        cluster[3].append(maps[(mappings[x])])
    elif i>144 and i<=165:
        cluster[4].append(maps[(mappings[x])])
    elif i>165 and i<=174:
        cluster[5].append(maps[(mappings[x])])
    elif i>174 and i<=206:
        cluster[6].append(maps[(mappings[x])])
    else:
        cluster[7].append(maps[(mappings[x])])
for i in range(8):
   cluster[i] = [np.array(x) for x in cluster[i]]
   cluster[i] = np.vstack(cluster[i])
   cluster[i] = sc.inverse_transform(cluster[i])
   cluster[i] = np.insert(cluster[i], 1, 1, axis=1)
result = np.concatenate((cluster[0], cluster[1], cluster[2], cluster[3], cluster[4], cluster[5],
                         cluster[6], cluster[7]), axis = 0)

#writes the result into a csv file
np.savetxt('result.csv', result, delimiter = ',')