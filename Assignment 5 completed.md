
# Programming Assignment 05 – Cluster Analysis


## KMeans Clustering (k=5)

**Question 1:**  
**First centroid x-coordinate:** `1.72`

**Question 2:**  
**First centroid y-coordinate:** `0.61`

**Question 3:**  
**Inertia after training:** `13.22`

### Code

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load and prepare the data
healthy = pd.read_csv('healthy_lifestyle.csv')
X = healthy[['sunshine_hours', 'happiness_levels']].dropna()

# Standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=5, init='random', n_init=10, random_state=123, algorithm='elkan')
kmeans.fit(X)

# Output
centroid = kmeans.cluster_centers_
print("Centroids:", np.round(centroid, 4))

inertia = kmeans.inertia_
print("Inertia:", round(inertia, 4))
```

---

## Hierarchical Clustering (k=3)

**Question 4:**  
**Label for first example (row 0):** `0`

**Question 5:**  
**Happiness level for first example:** `7.44`

**Question 6:**  
**First element of linkage matrix (SciPy):** `39`

### Code

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

healthy = pd.read_csv('healthy_lifestyle.csv')
X = healthy[['sunshine_hours', 'happiness_levels']].dropna()

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
print("First row label:", labels[0])

# Original happiness value
print("Happiness level for row 0:", healthy['happiness_levels'].iloc[0])

# SciPy linkage
clustersHealthyScipy = linkage(X, method='ward')
print("First five rows of linkage matrix:
", np.round(clustersHealthyScipy[:5, :], 0))
```

---

## DBSCAN (eps=1, min_samples=8, sample=500)

**Question 7:**  
**Number of core points:** `494`

**Question 8:**  
**Number of outliers:** `2`

### Code

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and sample data
data = pd.read_csv('customer_personality.csv').sample(500, random_state=123)
X = data[['Fruits', 'Meats']]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=8)
dbscan.fit(X)

# Outputs
print("Labels:", dbscan.labels_)
print("Number of core points:", len(dbscan.core_sample_indices_))
print("Number of outliers:", list(dbscan.labels_).count(-1))
```
