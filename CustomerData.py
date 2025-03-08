#!/usr/bin/env python
# coding: utf-8

# In[5]:


############### Unsuopervised Learning ############


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering 


from sklearn.metrics import silhouette_score , calinski_harabasz_score , davies_bouldin_score
from sklearn.cluster import KMeans , MiniBatchKMeans, MeanShift, estimate_bandwidth , AffinityPropagation , AgglomerativeClustering , DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from kneed import KneeLocator
import tkinter as tk

import warnings
warnings.simplefilter("ignore")  


get_ipython().system('pip install kneed')



# In[7]:


data = pd.read_csv(r'C:\Users\iran\Desktop\dataset.exel\Customer_Data.csv')
data


# In[8]:


df = pd.DataFrame(data)
df


# In[9]:


df.describe().T


# In[10]:


#### Drop  CUST_ID  #####
df1 = df.drop(columns = ["CUST_ID"])
df1


# In[11]:


df1.info()


# In[12]:


df1.isnull().sum()


# In[13]:


df1["CREDIT_LIMIT"]


# In[14]:


df2 = df1.dropna()


# In[15]:


df2.describe().T


# In[16]:


df2.columns


# In[17]:


columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
       'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
       'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
       'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
       'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT',
       'TENURE']


# In[18]:


fig=plt.figure(figsize=(35,40))
for i,col in enumerate(df2.drop(columns='BALANCE').columns):
    ax=fig.add_subplot(4 , 4,i+1)
    sns.scatterplot(x=col, y ='BALANCE' ,data=df2 , color = 'green')


# In[19]:


df2.hist(bins=30,figsize=(20,15), color="yellow")
plt.show()


# In[20]:


sns.pairplot(df2)


# In[21]:


plt.figure(figsize =(20 , 10))
hm = sns.heatmap(df2.corr() , annot=True )
hm.set(title = "Crrelation matrix of the data")
plt.show()


# In[22]:


for i in columns :

    plt.figure(figsize=(6,3))
    sns.histplot(df2[i], kde=True, color='r')
    plt.xlabel(i)


# In[23]:


i = 0 
while i < 16 :
    fig = plt . figure(figsize=[16 , 3])
    plt . subplot(1 ,3 ,1)
    sns . boxplot(x =columns[i] , data =df2 ,color='darkorchid')
   

    plt . xlabel(columns[i] )
    i+=1
    if i== 16 :
        break
    plt . subplot(1 , 3 , 2)
    sns . boxplot(x =columns[i] , data = df2 ,color='darkorchid' )
    i += 1


# In[24]:


df2.boxplot(rot=90 , figsize=(30,20))


# In[25]:


end1 = df2[(df2['INSTALLMENTS_PURCHASES'] > 20000)].index
end2 = df2[(df2['CASH_ADVANCE'] > 40000)].index
end3 = df2[(df2['MINIMUM_PAYMENTS'] > 40000)].index
end4 = df2[(df2['ONEOFF_PURCHASES'] > 30000)].index
end5 = df2[(df2['CASH_ADVANCE_TRX'] > 70)].index
end6 = df2[(df2['CREDIT_LIMIT'] > 25000)].index
end7 = df2[(df2['PAYMENTS'] > 42000)].index
end8 = df2[(df2['PURCHASES'] > 45000)].index
end9 = df2[(df2['BALANCE'] > 17000)].index


# In[26]:


print(end1)
print(end2)
print(end3)
print(end4)
print(end5)
print(end6)
print(end7)
print(end8)
print(end9)


# In[27]:


df3 = df2.copy()


# In[28]:


df3.drop([5260,2159,4376, 4462, 5657, 5830, 5968, 7132 ,501, 550, 1604, 3937,
         542, 1913,  3545, 5116, 5287, 5319, 8315,970, 4905, 7046, 4220,122, 138, 4140] ,inplace=True)
df3.reset_index(inplace=True, drop=True)
df3


# In[29]:


################ K_Mean Model ############


# In[30]:


kmean_set = {"init" : "random" , "n_init" : 10 , "max_iter" : 300 , "random_state" : 42}


# In[31]:


scaler = StandardScaler()
scaler_features = scaler.fit_transform(df3)


# In[32]:


list = []
for k in range(1,20):
    kmeans = KMeans(n_clusters=k, **kmean_set)
    kmeans.fit(scaler_features)
    list.append(kmeans.inertia_)


# In[33]:


plt.figure(figsize =(20 , 15))
plt.style.use("fivethirtyeight")
plt.plot(range(1,20) , list , 'o')
plt.plot(range(1,20), list,'-' , alpha = 0.5)
plt.xticks(range(1,20))
plt.xlabel("number of clusters")
plt.ylabel("inertia")
plt.show()


# In[34]:


k1 = KneeLocator(range(1,20) , list , curve="convex" , direction = "decreasing")
k1.elbow


# In[35]:


plt.figure(figsize =(20 , 15))
plt.style.use("fivethirtyeight")
plt.plot(range(1,20), list , 'o')
plt.plot(range(1,20), list,'-' , alpha = 0.5)
plt.xticks(range(1,20)) 
plt.xlabel(" number of clusters")
plt.ylabel("inertia")
plt.axvline(x=k1.elbow, color='darkred', label="anxline - full height", ls='--')
            
plt.show()


# In[36]:


kmeans = KMeans(n_clusters=2, random_state=42)

kmeans.fit_predict(scaler_features)
centroids = kmeans.cluster_centers_

score = silhouette_score(scaler_features, kmeans.labels_, metric='euclidean')
# Print the score
print('Silhouetter Average Score: %.3f' % score)

plt.figure(figsize =(7,5))
plt.scatter(df3['PURCHASES'], df3['BALANCE'], c= kmeans.labels_.astype (float), s=50, alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,2], c = 'red', s=50)  
plt.show()


# In[37]:


kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit_predict(scaler_features)
centroids = kmeans.cluster_centers_

score = silhouette_score(scaler_features, kmeans.labels_, metric='euclidean')
# Print the score
print('Silhouetter Average Score: %.3f' % score)

plt.scatter(df3['PURCHASES'], df3['BALANCE'], c= kmeans.labels_.astype (float), s=50, alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,2], c = 'red', s=50)  
plt.show()


# In[38]:


from sklearn.metrics import silhouette_score

silhouette_coefficients = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, **kmean_set)
    kmeans.fit(scaler_features)
    score = silhouette_score(scaler_features, kmeans.labels_)
    silhouette_coefficients.append(score)
    
plt.figure(figsize =(20 , 15))    
plt.style.use ("fivethirtyeight")
plt.plot(range(2, 20), silhouette_coefficients)
plt.xticks(range(2, 20))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()   
####   silhouette_score claims the most appropriate number of clusters is 3


# In[39]:


kmeans = KMeans(n_clusters=4, random_state=42)

kmeans.fit_predict(scaler_features)
centroids = kmeans.cluster_centers_

score = silhouette_score(scaler_features, kmeans.labels_, metric='euclidean')
# Print the score
print('Silhouetter Average Score: %.3f' % score)

plt.scatter(df3["PURCHASES"], df3['BALANCE'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel ("PURCHASES")
plt.show()


# In[40]:


silhouette_coefficients =[]
for k in range(2,20):#1 is the worse
    kmeans=KMeans(n_clusters=k, **kmean_set)
    kmeans.fit(scaler_features)
    score= silhouette_score(scaler_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[41]:


silhouette_coefficients


# In[42]:


calinski_harabasz_coefficients =[]
for k in range(2,20):#1 is the worse
    kmeans=KMeans(n_clusters=k, **kmean_set)
    kmeans.fit(scaler_features)
    score= calinski_harabasz_score(scaler_features, kmeans.labels_)
    calinski_harabasz_coefficients.append(score)


# In[43]:


calinski_harabasz_coefficients


# In[44]:


davies_bouldin_coefficients =[]
for k in range(2,20):#1 is the worse
    kmeans=KMeans(n_clusters=k, **kmean_set)
    kmeans.fit(scaler_features)
    score= davies_bouldin_score(scaler_features, kmeans.labels_)
    davies_bouldin_coefficients.append(score)


# In[45]:


davies_bouldin_coefficients


# In[46]:


###########  GMM  ##################


# In[47]:


range_n_clusters = [2, 3, 4]

for num_clusters in range_n_clusters:
    
    
    GMM = GaussianMixture(n_components=num_clusters, random_state=0, covariance_type="full")
    GMM.fit(scaler_features)
    
    
    # silhouette score
    silhouette_avg = silhouette_score(scaler_features, GMM.predict(scaler_features))
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[48]:


range_n_clusters = [2, 3, 4]

for num_clusters in range_n_clusters:
    
    
    GMM = GaussianMixture(n_components=num_clusters, random_state=0, covariance_type="full")
    GMM.fit(scaler_features)
    
    
    # calinski_harabasz score
    calinski_harabasz_avg = calinski_harabasz_score(scaler_features, GMM.predict(scaler_features))
    print("For n_clusters={0}, the calinski_harabasz score is {1}".format(num_clusters, calinski_harabasz_avg))


# In[49]:


range_n_clusters = [2, 3, 4]

for num_clusters in range_n_clusters:
    
    
    GMM = GaussianMixture(n_components=num_clusters, random_state=0, covariance_type="full")
    GMM.fit(scaler_features)
    
    
    # davies_bouldin score
    davies_bouldin_avg = davies_bouldin_score(scaler_features, GMM.predict(scaler_features))
    print("For n_clusters={0}, the davies_bouldin score is {1}".format(num_clusters,davies_bouldin_avg))


# In[50]:


silhouette_scores = []

for n_cluster in range(2, 20):
    silhouette_scores.append( 
        silhouette_score(scaler_features, KMeans(n_clusters = n_cluster).fit_predict(scaler_features))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
plt.figure(figsize =(20 , 15))
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()


# In[51]:


calinski_harabasz = []

for n_cluster in range(2, 20):
    calinski_harabasz.append( 
        calinski_harabasz_score(scaler_features, KMeans(n_clusters = n_cluster).fit_predict(scaler_features))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
plt.figure(figsize =(20 , 15))
plt.bar(k, calinski_harabasz) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('calinski_harabasz Score', fontsize = 10) 
plt.show()


# In[52]:


davies_bouldin = []

for n_cluster in range(2, 20):
    davies_bouldin.append( 
        davies_bouldin_score(scaler_features, KMeans(n_clusters = n_cluster).fit_predict(scaler_features))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
plt.figure(figsize =(20 , 15))
plt.bar(k, davies_bouldin) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('davies_bouldin Score', fontsize = 10) 
plt.show()


# In[53]:


################# Hierarichical clustering ###############


# In[54]:


# Dendrogram for Hierarchical Clustering
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(20,14))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(scaler_features, method='ward'))


# In[55]:


hierarchical_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
hierarchical_clustering.fit(scaler_features)
hierarchical_labels = hierarchical_clustering.labels_


# In[56]:


hierarchicaldf3_Cluster = df3.copy()
hierarchical_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
hierarchical_clustering.fit(scaler_features)
hierarchical_labels = hierarchical_clustering.labels_
hierarchicaldf3_Cluster['Cluster'] = hierarchical_labels
hierarchicaldf3_Cluster


# In[57]:


silhouette_avg = silhouette_score(scaler_features, hierarchical_clustering.labels_ )
print("\nthe silhouette score is : " , silhouette_avg)
    
calinski_score_avg = calinski_harabasz_score(scaler_features, hierarchical_clustering.labels_ )
print("\nthe calinski_score score is : " , calinski_score_avg)
    
davies_bouldin_Score_avg = davies_bouldin_score(scaler_features, hierarchical_clustering.labels_ )
print("\nthe davies_bouldin score is : " , davies_bouldin_Score_avg)
    


# In[58]:


############### Mean Shift ##############


# In[59]:


bandwidth = estimate_bandwidth(scaler_features,n_samples=300, n_jobs=-1)


# In[60]:


Mean_shift = MeanShift(bandwidth=bandwidth,bin_seeding=True)
Mean_shift.fit(scaler_features)


# In[61]:


Mean_shift_Clusters = Mean_shift.labels_
np.unique(Mean_shift_Clusters)


# In[62]:


print(f'silhouette_score : {silhouette_score(scaler_features ,Mean_shift.labels_)}')
print(f'calinski_harabasz_score : {calinski_harabasz_score(scaler_features ,Mean_shift.labels_)}')
print(f'davies_bouldin_score : {davies_bouldin_score(scaler_features ,Mean_shift.labels_)}')


# In[63]:


################ Affinity Propagtion ###############


# In[64]:


affinity_propagation_set = {
    "damping": 0.9,
    "max_iter": 200,
    "convergence_iter": 15,
    "affinity": 'euclidean',
    "preference": None,
    "random_state": 42
}


# In[65]:


affinity_propagation_clustering = AffinityPropagation(**affinity_propagation_set)
affinity_propagation_clustering.fit(scaler_features)


# In[66]:


affinity_propagation_labels = affinity_propagation_clustering.labels_

# Get the number of clusters identified by AffinityPropagation
num_clusters_affinity_propagation = len(np.unique(affinity_propagation_labels))

print("Number of clusters identified by AffinityPropagation:", num_clusters_affinity_propagation)


# In[67]:


print(f'silhouette_score : {silhouette_score(scaler_features ,affinity_propagation_clustering.labels_)}')
print(f'calinski_harabasz_score : {calinski_harabasz_score(scaler_features ,affinity_propagation_clustering.labels_)}')
print(f'davies_bouldin_score : {davies_bouldin_score(scaler_features ,affinity_propagation_clustering.labels_)}')


# In[68]:


################## DBSCAN ###################


# In[69]:


db = DBSCAN(eps=4 , min_samples=34).fit(scaler_features)


# In[70]:


silhouette_avg = silhouette_score(scaler_features, db.labels_ )
print("\nthe silhouette score is : " , silhouette_avg)

calinski_score_avg = calinski_harabasz_score(scaler_features, db.labels_ )
print("\nthe calinski_score score is : " , calinski_score_avg)

davies_bouldin_Score_avg = davies_bouldin_score(scaler_features, db.labels_ )
print("\nthe davies_bouldin score is : " , davies_bouldin_Score_avg)


# In[71]:


plt.figure (figsize = (10 , 5))
plt.scatter(df3['PURCHASES'], df3['BALANCE'], c= db.labels_.astype (float), s=50, alpha=0.5)
plt.show()


# In[72]:


plt.figure (figsize = (10 , 5))
plt.scatter(df3['BALANCE'], df3['PURCHASES'], c= db.labels_.astype (float), s=50, alpha=0.5)
plt.show()


# In[73]:


#################  Heirarichicaal is the best model 
#################  We choose the Heirarichicaal With Outliers as the final model


# In[ ]:




