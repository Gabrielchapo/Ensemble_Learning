# Map of the code

cyberbully.ipynb contains the home-made implementation


# Ensemble_Learning

Random Forest is one of the most widely used supervised machine learning algorithms for classification and regression problems. It builds Decision Trees on different samples of the training dataset and puts them together to vote for the class in the case of classification and the mean in the case of regression. The idea is to combine several weak learners, here the decision trees, to improve the robustness and accuracy of predictions in the Random Forest.
This algorithm has many qualities. In addition to being able to process data containing continuous and discrete variables, it is an easily interpretable algorithm, unlike deep learning algorithms which are real black boxes.

## Method & Implementation:
Each algorithm is implemented in an Object-oriented programming way. You can find the method fit, predict and score.
### Decision Tree :
Several parameters can be chosen during the instantiation:
The maximum depth of the tree. (max_depth)
The minimum number of samples required to split an internal node. (min_samples_split)
The minimum number of samples required to be at a leaf node. (min_samples_leaf)
To increase speed and prevent overfitting, we do not calculate the impurity for each node but for each decile. (splitting)
We first initialise the decision tree by positioning ourselves at the root of it, and we look at the entire dataset. From this starting point and recursively, we will:
Iterate through each feature. If the parameter splitting is False, we will iterate through each value of each observation for each feature, taking care to remove duplicates. If the parameter splitting is True, we look at each decile of each feature (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%) to make the split.
We will split the dataset into two groups according to each of these previously generated values, then compute the Gini index, determining the impurity of these two groups. As we go along, we only take the value generating the lowest impurity.
Once the iteration is completed, we have the value for a feature generating the lowest impurity. We split the dataset in two following this value, then we repeat all these steps for each of the two groups generated until we reach the maximum depth, the minimum number of samples required to split a node or the minimum samples required to be at a leaf node.

The Gini score tells us how good a split is by how mixed the classes are in the two groups created by the split. For example, 0 is the perfect score, meaning the group is pure, 0.5 is the worst case in a two classes scenario.

The decision tree takes the form of a dictionary, which will be easily browsed during inference, to determine to which class an observation belongs.

The parameter not only allows us to drastically reduce the execution time (scanning each observation of a dataset versus scanning 10 values: the deciles) but also gives us mostly better results. 

### Random Forest :

As we saw earlier, the Random Forest is a set of Decision Trees. In our implementation, we find the same parameters as in our implementation of the Decision Tree, as well as the parameter n_estimators which indicates the number of Decision Trees desired.

During training and for each Decision Tree, a random sample of data points is generated for training. We perform Bootstrapping, which means that the samples are drawn with replacement and two thirds of the Dataset. Some samples will therefore be used multiple times in a single tree. We do this in order to have each tree trained on different samples, although individually and potentially having a high variance, put together in a random forest with a lower variance, without increasing the bias.

### AdaBoost :

Another solution that we chose to implement from scratch is Adaboost. This is a boosting technique mainly used for classification tasks. 
Our algorithm takes into argument two hyper-parameters which are the depth of the trees and the number of classifiers. 
Adaboost is constructed based on the boosting technique, the idea is to train many weak models in order to build a strong model. 
After fitting all the models we combined to make the final prediction. The models are pondered based on their error rate in order to favor a more performant model.

## Results :

We tested these two algorithms in the context of the project of classifying hate messages into different categories: age, ethnicity, gender, not_cyberbullying, other_cyberbullying, religion.

We used a SkipGram model to create a representation of each word, in the form of an array of size 25, and then a representation of each message by calculating the averages of the words included in it.

We wanted to compare our results with the implementation proposed by sklearn : tree and ensemble.RandomForestClassifier. The dataset has been split on one side for the training (70% of the dataset) and on the other side for the test part (30% of the dataset).

On the parameter side, we opted for a max_depth of 10, a min_samples_split of 6, a min_samples_leaf of 1 and an _estimators of 10.

Our results are close to those obtained with sklearn, but the execution time differs greatly:
- Our Decision Tree takes 19.3 seconds to train, for a score on the test set of 0.664.
- The sklearn Decision Tree takes 0.8 seconds to train but produces a score of 0.659.
- Our Random Forest takes 156.4 seconds to train, for a score on the test set of 0.708.
- The sklearn Random Forest takes 1.7 seconds to train but produces a score of 0.709.

## Conclusion :

Our first results are encouraging, we proved that the random forest is much more efficient than the decision tree, but we still have a long way to go on the optimization of the code and the method used to get closer to the execution speed of sklearn.
