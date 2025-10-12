# Decision Tree Classifier by "hand"

## What is a decision tree
A DT is a model that tries to make logic decisions based on question like:

- "The value of HP is lower or equal to 50 (**threshold**)?"
  - If yes, goes to the left
  - If not, goes to the right

Each **node** of the tree asks a question about a **feature**, and at the end (**on the leafs**), are the **final class** (ex.: ```"Legendary = True"``` or ```"False"```).

## Calculate Entropy

```calculate_entropy(y)```

This calculates the entropy from the y vector given, that is, the degree of disorder or mixing between classes.

- If there's only 1 class, ```entropy=0)``` (Pure).
- If there's like a 50/50 between 2 classes, the ```entropy=1)``` (maximum uncertainly).

``` vals, counts = np.unique(y, return_counts=True)``` - Founds the **unique classes** (ex.: ```[True, False]```) and how many times each one shows.

For example:

``` python
y = [True, False, True, False]
vals = [True, False]
counts = [0.5, 0.5]
```

```probs = counts / counts.sum()``` - Convert the counts into probabilities, on this examples the probs are ```[0.5, 0.5]```.

Then we return the entropy by using the function entropy from the scipy.stats: ```return stats.entropy(probs)```.
This function calculates how mixed the labels are based on the probabilities we just found.

## Finding the best threshold for a feature
```best_threshold_for_feature(X, y, feature)```

Finds the best cut-off point (**threshold**) in a certain **feature**, to divide the data and make classes more "pure".

For example, taking the feature ```HP```, that assigning imaginary values has ```[10, 20, 30, 40, 50]```, and the classes are, respectively, ```[False, False, True, True, True]```.

The code will attempt to cut between unique values, such as:

- Between 10 and 20 --> ```thr = 15```
- Between 20 and 30 --> ```thr = 25```
- Between 30 and 40 --> ```thr = 35```
- Between 40 and 50 --> ```thr = 45```

At each ```thr```(threshold), divide the data:
- Left: values ```<= thr```
- Right: values ```> thr```

Then it measures the **entropy before and after** the cut and calculates how much it has improved (**information gain**).

*In the code:*

``` python
x = X[feature].to_numpy()
y = np.asarray(y)
```

Convert the data into numpy arrays (to make comparison faster).

``` python
uniq = np.unique(x)
if uniq.size <= 1
    return None, 0.0
```

Removes repeated values from the feature.
If after that there’s only **one unique value**, it means all examples have the same number for this feature (for example, all ```HP = 50```).
In that case, there’s no possible threshold to split.

``` python
cands = (uniq[:-1] + uniq[1:]) / 2.0
```

"Candidates", generate the **midpoints** between consecutive values, for example: ```[10, 20, 30] --> (10+20)/2=15, (20+30)/2=25 --> [15, 25]```

``` python
base_e = calculate_entropy(y)
```

Calculate entropy before dividing.

```
for thr in cands:
    left_mask = x <= thr
    right_mask = ~left_mask
```
Creates boolean masks:

- ```left_mask``` --> array of ```True/False``` indicating each one goes to the left.
- ```right_mask``` --> the inverse of the left mask

``` python
yl = y[left_mask]
yr = y[right_mask]
```
We take the classes from the left and the right.

``` python
el = calculate_entropy_y(yl)
er = calculate_entropy_y(yr)
```
And we calculate the entropy of each side.

``` python
weighted = (yl.size / n) * el + (yr.size / n) * er
gain = base_e - weighted
```

So that we can calculate the **information gain** (how much the total mixture was reduced).

``` python
if gain > best_gain:
    best_gain = gain
    best_thr = thr
```

We keep the **threshold with the best** **information gain**, and we return both (```return best_thr, best_gain```).

## Majority class
```majority_class(y)```

Chooses the **most frequent class** (in case of a draw, choose the first one).

For example: ```[True, False, False, False] --> False```. We use this when the **node** is a **leaf**.

## Building the Tree
```build_tree(X, y, features, max_depth=None, depth=0)```

Creates the tree **recursively**:

1. Checks if it is a leaf.
2. If not, chooses the best feature + threshold.
3. Divides data and repeats.

*In the code:*

``` python
if np.unique(y).size == 1:
    return {"leaf": True, "class": y[0]}
```
If all classes are equal --> pure node --> **leaf**.

``` python
if (max_depth is not None and depth >= max_depth):
    return {"leaf": True, "class": majority_class_y(y)}
```
If maximum depth has been reached --> **Stops here**.

``` python
best_feat = None
best_gain = -1.0
best_thr = None
```
Initialize the variables. ```best_gain``` starts at ```-1.0``` to ensure that any positive gain replaces it.

``` python
for feature in features:
    thr, gain = best_threshold_for_feature(X, y, feature)
    if gain > best_gain:
        best_gain = gain
        best_feat = feature
        best_thr = thr
```
Test **all the features**, see which one has the **best gain**, and keep the best one.

``` python
if best_thr is None or best_gain <= 0.0:
    return {"leaf": True, "class": majority_class(y)}
```
If there has been no improvement --> useless node --> make a leaf with the majority class.

``` python
left_mask = X[best_feat].to_numpy() <= best_thr
right_mask = ~left_mask
if not left_mask.any() or not right_mask.any():
    return {"leaf": True, "class": majority_class(y)}
```
Split the data between left and right based on the best feature and threshold.
If one side is empty --> useless node --> make a leaf with the majority class.

``` python
left_child = build_tree(X[left_mask], y[left_mask], features, max_depth, depth + 1)
right_child = build_tree(X[right_mask], y[right_mask], features, max_depth, depth + 1)
```

Call the function again (**recursion**) to build both sides of the tree.

``` python
return {
        "leaf": False,
        "feature": best_feat,
        "threshold": float(best_thr),
        "left": left_child,
        "right": right_child
    }
```
Returns the complete node.

## Predict One Sample
```predict_one(tree, row)```

Traverse the tree to a **row** until you reach the final class.

For example:

``` python
[HP <= 80]
  -> if 70 <= 80 -> goes left
```

*In the code:*

``` python
while not node["leaf"]:
    feature = node["feature"]
    thr = node["threshold"]
    node = node["left"] if row[feature] <= thr else node["right"]
return node["class"]
```

## Predict Tree
```predict_tree(tree, X)```

Predicts all the rows in ```X``` using the tree.
``` python
return np.array([predict_one(tree, X.iloc[i]) for i in range(X.shape[0])])
``` 
Uses a list comprehension to call ```predict_one``` for each row. ```iloc``` is used to access rows by their integer index, and ```X.shape[0]``` gives the number of rows in the DataFrame.

## Fit Tree
```fit_tree(X_train, y_train, features, max_depth=None)```

Trains the tree using the training data.

``` python
return build_tree(X_train, y_train, features, max_depth=max_depth)
```

## Decision Tree Classify
```dt_classify(X_train, y_train, X_test, features, max_depth=None)```

Combines everything:
1. Fit the tree (```fit_tree```)
2. Predict on the test set (```predict_tree```)
3. Return the results (```array``` with True/False)

``` python
tree = fit_tree(X_train, y_train, features, max_depth=max_depth)
return predict_tree(tree, X_test)
```

## Print Tree
```print_tree(tree, indent="", decimals=2```

Prints the tree in a human-readable format. For example:

``` python
[HP <= 78.50]
  [Speed <= 114.50]
    -> class: False
  [Speed >  114.50]
    -> class: True
```
