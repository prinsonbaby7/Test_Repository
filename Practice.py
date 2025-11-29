#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:26:59 2025

@author: prinsonbaby
"""

import pandas as pd
df = pd.read_excel(r"/Users/prinsonbaby/Desktop/Data Science360/Practice/Clustering_Dataset.xlsx")
df.info()
df.describe()
df.isnull().sum()
df = df.dropna()
print(df)
df = df.loc[df['Annual_Income (k$)'] >= 0]
print(df)
df = df.drop_duplicates()

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(figsize=(10, 6), bins=30)
plt.show()

sns.boxplot(data=df[['Age', 'Annual_Income (k$)', 'Spending_Score (1-100)']])
plt.show()

sns.scatterplot(data=df, x="Annual_Income (k$)", y="Spending_Score (1-100)")
plt.title("Income vs Spending Score")
plt.show()

from sklearn.cluster import KMeans

X = df[["Annual_Income (k$)", "Spending_Score (1-100)"]]
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Scatterplot for Clustering
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Annual_Income (k$)"], y=df["Spending_Score (1-100)"], hue=df["Cluster"], palette="viridis")
plt.title("Customer Segmentation")
plt.show()
