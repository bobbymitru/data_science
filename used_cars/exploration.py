#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:34:36 2020

@author: rmitru
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

data = pd.read_csv('vehicles.csv', index_col='id')

# DATA PROCESSING

# Drop duplicates
data.drop_duplicates(inplace=True)

# Outliers
data.drop(data[data.price < 800].index, axis=0, inplace=True)
data.drop(data[data.year > 2020].index, axis=0, inplace=True)
data.dropna(axis=0, subset=['year'], inplace=True)
data.dropna(axis=0, subset=['odometer'], inplace=True)

# year / price
fig, ax = plt.subplots()
ax.scatter(x=data.year, y=data.price)
plt.ylabel('price')
plt.xlabel('year')

data.drop(data[data.year < 1750].index, inplace=True)
data.drop(data[data.price > 700000].index, inplace=True)
data.drop(data[(data.year < 1980) & (data.price > 600000)].index, inplace=True)

# odometer / price
fig, ax = plt.subplots()
ax.scatter(x=data.odometer, y=data.price)
plt.ylabel('price')
plt.xlabel('odometer')

data.drop(data[(data.odometer > 500000) & (data.price > 100000)].index, inplace=True)

# 2) Target variable
fig, ax = plt.subplots(figsize=(10, 7))
sns.distplot(data.price, fit=norm)
(mu, sigma) = norm.fit(data.price)
plt.legend([f'Normal dist. ($\mu=${mu} and $\sigma=${sigma})'], loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')

fig = plt.figure()
res = stats.probplot(data.price, plot=plt)
plt.show()

skewness = data.price.skew()
kurtosis = data.price.kurt()

# 3) Features engineering
# Missing data
data.isnull().sum() / data.shape[0] * 100

# Drop features that have more that 60% null values
thresh = data.shape[0] * .6
data.dropna(thresh=thresh, axis=1, inplace=True)

# Drop observations that have more than 90% null values
thresh = data.shape[1] * .9
data.dropna(thresh=thresh, axis=0, inplace=True)

# Drop useless features
data.drop(['url', 'region', 'region_url', 'image_url', 'description', 'state', 'lat', 'long'], axis=1, inplace=True)

# Drop observations with null 'manufacturer' feature (can't fill these data in appropriate way)
data.dropna(axis=0, subset=['manufacturer'], inplace=True)

# Drop observations with null 'model' feature (can't fill these data in appropriate way)
data.dropna(axis=0, subset=['model'], inplace=True)

# Data correlation
corrmat = data.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)

# Imputting missing values
#Process vin feature
data.vin = data.vin.apply(lambda x: 'no_vin' if pd.isnull(x) else 'has_vin')

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=data.vin, y=data.price)
plt.ylabel('price')
plt.xlabel('vin')


# Fill condition feature
temp = data.dropna(axis=0, subset=['condition'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.condition, y=temp.price)
plt.ylabel('price')
plt.xlabel('condition')

data.condition.value_counts()

condition1 = (data.odometer == 0)
condition2 = (data.price > 220000)
condition3 = (data.price > 50000) & (data.price <= 220000)
condition4 = (data.price > 0) & (data.price <= 50000)

data.loc[condition1, 'condition'] = data.loc[condition1, 'condition'].fillna('new')
data.loc[condition2, 'condition'] = data.loc[condition2, 'condition'].fillna('excellent')
data.loc[condition3, 'condition'] = data.loc[condition3, 'condition'].fillna('good')
data.loc[condition4, 'condition'] = data.loc[condition4, 'condition'].fillna('like_new')

# Fill cylinders feature
temp = data.dropna(axis=0, subset=['cylinders'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.cylinders, y=temp.price)
plt.ylabel('price')
plt.xlabel('cylinders')

data.cylinders.value_counts()

condition1 = (data.price > 150000)
condition2 = (data.price > 100000) & (data.price <= 150000)
condition3 = (data.price > 50000) & (data.price <= 100000)
condition4 = (data.price > 0) & (data.price <= 50000)

data.loc[condition1, 'cylinders'] = data.loc[condition1, 'cylinders'].fillna('8 cylinders')
data.loc[condition2, 'cylinders'] = data.loc[condition2, 'cylinders'].fillna('6 cylinders')
data.loc[condition3, 'cylinders'] = data.loc[condition3, 'cylinders'].fillna('4 cylinders')
data.loc[condition4, 'cylinders'] = data.loc[condition4, 'cylinders'].fillna('5 cylinders')

# Fill fuel feature
temp = data.dropna(axis=0, subset=['fuel'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.fuel, y=temp.price)
plt.ylabel('price')
plt.xlabel('fuel')

data.fuel.value_counts()

condition1 = (data.price > 200000)
condition2 = (data.price > 100000) & (data.price <= 200000)
condition3 = (data.price > 0) & (data.price <= 100000)

data.loc[condition1, 'fuel'] = data.loc[condition1, 'fuel'].fillna('gas')
data.loc[condition2, 'fuel'] = data.loc[condition2, 'fuel'].fillna('diesel')
data.loc[condition3, 'fuel'] = data.loc[condition3, 'fuel'].fillna('other')

# Fill title_status feature
temp = data.dropna(axis=0, subset=['title_status'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.title_status, y=temp.price)
plt.ylabel('price')
plt.xlabel('title_status')

data.title_status.value_counts()

condition1 = (data.price > 100000)
condition2 = (data.price > 50000) & (data.price <= 100000)
condition3 = (data.price > 0) & (data.price <= 50000)

data.loc[condition1, 'title_status'] = data.loc[condition1, 'title_status'].fillna('clean')
data.loc[condition2, 'title_status'] = data.loc[condition2, 'title_status'].fillna('salvage')
data.loc[condition3, 'title_status'] = data.loc[condition3, 'title_status'].fillna('rebuilt')

# Fill transmission feature
temp = data.dropna(axis=0, subset=['transmission'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.transmission, y=temp.price)
plt.ylabel('price')
plt.xlabel('transmission')

data.transmission.value_counts()

condition1 = (data.price > 200000)
condition2 = (data.price > 100000) & (data.price <= 200000)
condition3 = (data.price > 0) & (data.price <= 100000)

data.loc[condition1, 'transmission'] = data.loc[condition1, 'transmission'].fillna('automatic')
data.loc[condition2, 'transmission'] = data.loc[condition2, 'transmission'].fillna('manual')
data.loc[condition3, 'transmission'] = data.loc[condition3, 'transmission'].fillna('other')

# Fill drive feature
temp = data.dropna(axis=0, subset=['drive'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.drive, y=temp.price)
plt.ylabel('price')
plt.xlabel('drive')

data.drive.value_counts()

condition1 = (data.price > 200000)
condition2 = (data.price > 150000) & (data.price <= 200000)
condition3 = (data.price > 0) & (data.price <= 150000)

data.loc[condition1, 'drive'] = data.loc[condition1, 'drive'].fillna('4wd')
data.loc[condition2, 'drive'] = data.loc[condition2, 'drive'].fillna('rwd')
data.loc[condition3, 'drive'] = data.loc[condition3, 'drive'].fillna('fwd')

# Fill type feature
temp = data.dropna(axis=0, subset=['type'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.type, y=temp.price)
plt.ylabel('price')
plt.xlabel('type')

data.type.value_counts()

condition1 = (data.price > 170000)
condition2 = (data.price > 100000) & (data.price <= 170000)
condition3 = (data.price > 0) & (data.price <= 100000)

data.loc[condition1, 'type'] = data.loc[condition1, 'type'].fillna('SUV')
data.loc[condition2, 'type'] = data.loc[condition2, 'type'].fillna('sedan')
data.loc[condition3, 'type'] = data.loc[condition3, 'type'].fillna('pickup')

# Fill paint_color feature
temp = data.dropna(axis=0, subset=['paint_color'])

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x=temp.paint_color, y=temp.price)
plt.ylabel('price')
plt.xlabel('paint_color')

data.paint_color.value_counts()

condition1 = (data.price > 150000)
condition2 = (data.price > 100000) & (data.price <= 150000)
condition3 = (data.price > 0) & (data.price <= 100000)

data.loc[condition1, 'paint_color'] = data.loc[condition1, 'paint_color'].fillna('white')
data.loc[condition2, 'paint_color'] = data.loc[condition2, 'paint_color'].fillna('silver')
data.loc[condition3, 'paint_color'] = data.loc[condition3, 'paint_color'].fillna('black')

data.cylinders.value_counts()
data.cylinders = data.cylinders.apply(lambda x: str(x).replace('cylinders', '').strip())
data.cylinders = pd.to_numeric(data.cylinders, errors='coerce')

data.odometer.max()

data.price = np.log1p(data.price)



