
# Data Mining Assignment – Frequent Itemset Analysis

This assignment analyzes a small market basket dataset to identify frequent itemsets and evaluate a given association rule.

## Dataset Used
File: `transactions_data[1].txt`  
Each line represents a transaction containing items bought together.

---

## Question 1  
**Which itemset has the highest frequency for frequent 1-itemset?**  
**Answer:** `['Tortilla chips']`  
**Explanation:** Appears in 7 out of 10 transactions.

---

## Question 2  
**Which itemset has the least frequency for frequent 2-itemset?**  
**Answer:** `['Pita chips', 'Salsa']`  
**Explanation:** This pair appears only once.

---

## Question 3  
**What is the consequent support for the rule `['Queso'] -> ['Salsa', 'Tortilla chips']`?**  
**Answer:** `0.5`  
**Explanation:** The itemset `['Salsa', 'Tortilla chips']` appears in 5 of 10 transactions.

---

## Question 4  
**What is the Confidence for the rule `['Queso'] -> ['Salsa', 'Tortilla chips']`?**  
**Answer:** `0.75`  
**Explanation:** 3 out of 4 transactions that include Queso also include both Salsa and Tortilla chips.

---

## Question 5  
**What is the Lift for the rule `['Queso'] -> ['Salsa', 'Tortilla chips']`?**  
**Answer:** `1.5`  
**Explanation:** Confidence / Support of consequent = 0.75 / 0.5 = 1.5

---

## YOUR CODE

```python
from collections import Counter
from itertools import combinations

# Load the transactions
transactions = [
    line.strip().split(',') for line in open("transactions_data[1].txt").readlines()
]

# 1-itemsets
item_counts = Counter()
for t in transactions:
    item_counts.update(t)
most_frequent_1_itemset = item_counts.most_common(1)
print("Most frequent 1-itemset:", most_frequent_1_itemset)

# 2-itemsets
itemset_2_counts = Counter()
for t in transactions:
    t_unique = list(set(t))  # avoid counting duplicates in same basket
    itemset_2_counts.update([tuple(sorted(pair)) for pair in combinations(t_unique, 2)])
least_frequent_2_itemset = itemset_2_counts.most_common()[-1]
print("Least frequent 2-itemset:", least_frequent_2_itemset)

# Rule metrics: ['Queso'] -> ['Salsa', 'Tortilla chips']
def support(itemset):
    return sum(1 for t in transactions if all(item in t for item in itemset)) / len(transactions)

support_antecedent = support(['Queso'])
support_consequent = support(['Salsa', 'Tortilla chips'])
support_both = support(['Queso', 'Salsa', 'Tortilla chips'])

confidence = support_both / support_antecedent
lift = confidence / support_consequent

print("Support of Consequent:", round(support_consequent, 3))
print("Confidence:", round(confidence, 3))
print("Lift:", round(lift, 3))
```
