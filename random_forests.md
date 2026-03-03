# 🌲 Random Forest: Internal Mechanics & Mathematical Foundations

## 1. High-Level Architecture 🏗️
A Random Forest is an ensemble learning method that builds multiple decision trees and aggregates their results. It prevents overfitting by introducing randomness into the tree-building process.

* **Bootstrapping:** The algorithm creates multiple subsets of the training data by sampling with replacement. Each tree is trained on one of these subsets.
* **Random Subspacing:** At every node split, a tree evaluates only a random subset of features ($m$), rather than all available features. 
* **Aggregation:** Predictions from all fully grown trees are combined to form the final model output.
    * *Classification:* Majority voting (the mode).
    * *Regression:* Average of all predictions.



[Image of Random Forest structure]


---

## 2. Classification Mechanics: Measuring Purity 🧲
For categorical targets (e.g., Spam vs. Not Spam), the trees split data to maximize the "purity" (homogeneity) of the resulting child nodes. A node with only one class is perfectly pure (impurity = 0).

### Loss Functions
The algorithm evaluates splits using one of two primary metrics:

* **Gini Impurity** 📉: Measures the probability of misclassifying a randomly chosen element. The tree searches for the feature split that results in the lowest Gini score.
    $$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
    *(Where $C$ is the total classes and $p_i$ is the probability of picking class $i$)*
* **Entropy** 🌀: Measures the thermodynamic "disorder" or information density of a node. 
    $$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$



---

## 3. Regression Mechanics: Measuring Variance 📏
For continuous numerical targets (e.g., House Prices), trees split data to group similar numerical values together. The goal is to minimize the variance within each child node.

### Loss Function
* **Mean Squared Error (MSE)** 🎯: The algorithm tests splits to find the one that causes the largest drop in MSE compared to the parent node.
    $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$
    *(Where $n$ is the number of samples, $y_i$ is the actual target value, and $\bar{y}$ is the node's average target value)*

---

## 4. The Ensemble Math: Variance Reduction 🧮
A single decision tree often suffers from high variance (overfitting). Random Forests solve this by averaging multiple uncorrelated trees.

The mathematical variance of the forest's average prediction is defined as:
$$\text{Variance} = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$$
*(Where $\rho$ is the correlation between trees, $\sigma^2$ is the variance of a single tree, and $B$ is the total number of trees)*

**Why the randomness works:**
1.  As the number of trees ($B$) increases, the second term ($\frac{1-\rho}{B} \sigma^2$) shrinks toward zero.
2.  Bootstrapping and random subspacing force the trees to look at different data and features, driving the correlation ($\rho$) toward 0. 
3.  When $\rho \approx 0$, the first term ($\rho \sigma^2$) vanishes, leaving the model with a drastically reduced overall variance.