# Linear Regression Implementation (Report)
## 1D Linear Regression (SLR)
Simple Linear Regression (SLR) is a statistical model used to find the best straight line that explains the relationship between two variables.  
The line has the form:

`y = a*x + b`

- `a` -> slope of the line (how much y changes for each unit of x)
- `b` -> intercept (the value of y when x = 0)

The goal is to find the values of `a` and `b` that minimize the total squared error between the predicted and actual values.

---

### Function: linearRegressionSLR(X, y)

Find the slope (`a`) and intercept (`b`) that best fit the data points (X, y).

#### Step by step

```python
x = np.asarray(X).ravel()
y = np.asarray(y).ravel()
```

- `np.asarray()` converts a list, pandas Series, or other data type into a NumPy array.  
  NumPy arrays are used because they allow efficient mathematical operations.

- `.ravel()` flattens the array into one single dimension.  
  For example:
  ```
  [[1], [2], [3]] -> [1, 2, 3]
  ```
  This ensures that even if the data was given as a column (2D), it becomes a 1D vector, which is required for the regression formulas.

---

```python
mx = x.mean()
my = y.mean()
```

- Calculates the mean (average) of x and y.
- These means are used to center the data around the average values before computing the slope.

---

```python
a = np.sum((x - mx)*(y - my)) / np.sum((x - mx)**2)
```

- `(x - mx)` and `(y - my)` subtract the mean from each element, giving how far each point is from the average.
- The numerator `np.sum((x - mx)*(y - my))` measures the covariance between x and y.
- The denominator `np.sum((x - mx)**2)` measures the variance of x.
- Dividing these two gives the slope `a`, representing how y changes when x increases.

---

```python
b = my - a*mx
```

- Once the slope is known, the intercept `b` is calculated using the average values of x and y.
- It represents where the line crosses the y-axis (the predicted y when x = 0).

---

```python
return a, b
```

- Returns both parameters of the model.

---

### Function: predictSLR(X, a, b)

Generate predictions using the learned model parameters.

```python
y_pred = a * X + b
```

- Applies the linear equation directly to all x values.
- If `X` is a NumPy array, this operation is vectorized — it computes all results at once, without loops.

```python
return y_pred
```

- Returns the array of predicted y values.

## Multiple Linear Regression (MLR)

Multiple Linear Regression (MLR) extends the same concept to **more than one input variable**.  
Instead of a line, the model fits a **plane** (in 2D) or a **hyperplane** (in higher dimensions) that best predicts the output.  
The general model is:

`y = w1*x1 + w2*x2 + ... + wn*xn + b`

- Each `wi` -> represents the contribution of feature `xi` to the target.  
- `b` -> intercept (the value of y when all features are zero).

The goal is the same: find the coefficients that minimize the sum of squared residuals between predicted and actual values.

---

### Function: linearRegressionMLR(X, y)

Compute all coefficients (`w`) for multiple features using the **Ordinary Least Squares (OLS)** method.

#### Step by step

```python
X = np.asarray(X)
y = np.asarray(y).ravel()
```

- Converts the features matrix `X` and target vector `y` into NumPy arrays.  
- Ensures that `y` is 1D, because NumPy operations expect `(n,)` shape, not `(n,1)`.

---

```python
XTX = np.dot(X.T, X)
```

- Calculates `Xᵀ * X`, the correlation between all pairs of features. 
- `np.dot()` performs matrix multiplication.
- The result is a square matrix `(n_features x n_features)` that represents how each variable relates to the others.

---

```python
XTy = np.dot(X.T, y)
```

- Computes `Xᵀ * y`, the correlation between each feature and the target variable.
- `np.dot()` again performs matrix multiplication.
- The result is a vector `(n_features,)` that shows how each feature relates to the target.

---

```python
w = np.dot(np.linalg.inv(XTX), XTy)
```

- Applies the **normal equation** to find the coefficients:  
  `(Xᵀ X)⁻¹ Xᵀ y`
- `np.linalg.inv(XTX)` computes the inverse of the matrix `Xᵀ X`.
- `np.dot(..., XTy)` multiplies the inverse with `Xᵀ y` to get the coefficients.
- This is the closed-form OLS solution that minimizes the total squared error.
- The result `w` is a vector of coefficients — one for each feature.

---

```python
return w
```

- Returns the array of learned weights.

---

### Function: predictMLR(X, w)

Predict the target values `y_pred` using the learned coefficients.

```python
y_pred = np.dot(X, w)
```

- Performs the matrix multiplication (`np.dot()`) between the features matrix `X` and the weights vector `w`.
- This computes the predicted values for all samples in one go.
  
```python
return y_pred
```

- Returns the vector of predictions.
