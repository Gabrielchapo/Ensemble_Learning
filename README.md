# Abstract

In This course project, we tackled the problem of text-based classification to classify tweets within six classes of cyberbullying.

Our main objective is to build a strong model, based on ensemble learning methods. we had two challenges, the first was to perform several NLP techniques to
extract features from texts dataset, and the second was to apply several classifiers, tuning them and then choosing the best model for this task.

We devided the projects in three tasks, then we applied several methods on each step:
* Text preprocessing: lemmatization and stemming, stop words ...
* Word/Sentence Embedding: Bag of words, TF IDF, Word2Vec 
* Classification: GridSearch on (Decision Trees, Bagging, Random forests, Boosting, Gradient Boosted Trees, AdaBoost)

At the end, we compared the accuracies of each couple of methods then we chose the combination (word embedding, classification) that have the best accuracy.

We performed each of the tasks above twice: first on a multi-class classification task, then in a binary classification task.



# Map of the code

cyberbully.ipynb contains the home-made implementation
cyberbullying_final_notebook.ipynb contains the final code for the best model the project 


# Ensemble_Learning

Random Forest is one of the most widely used supervised machine learning algorithms for classification and regression problems. It builds Decision Trees on different samples of the training dataset and puts them together to vote for the class in the case of classification and the mean in the case of regression. The idea is to combine several weak learners, here the decision trees, to improve the robustness and accuracy of predictions in the Random Forest.

## Method & Implementation:
Each algorithm is implemented in an Object-oriented programming way. You can find the method fit, predict and score.
### Decision Tree :
Several parameters can be chosen during the instantiation:
The maximum depth of the tree. (max_depth)
The minimum number of samples required to split an internal node. (min_samples_split)
The minimum number of samples required to be at a leaf node. (min_samples_leaf)
To increase speed and prevent overfitting, we do not calculate the impurity for each node but for each decile. (splitting)


### Random Forest :

As we saw earlier, the Random Forest is a set of Decision Trees. In our implementation, we find the same parameters as in our implementation of the Decision Tree, as well as the parameter n_estimators which indicates the number of Decision Trees desired.

### AdaBoost :

Another solution that we chose to implement from scratch is Adaboost. This is a boosting technique mainly used for classification tasks. 
Our algorithm takes into argument two hyper-parameters which are the depth of the trees and the number of classifiers. 
Adaboost is constructed based on the boosting technique, the idea is to train many weak models in order to build a strong model. 
After fitting all the models we combined to make the final prediction. The models are pondered based on their error rate in order to favor a more performant model.

## Results :

Our results are close to those obtained with sklearn, but the execution time differs greatly:
- Our Decision Tree takes 19.3 seconds to train, for a score on the test set of 0.664.
- The sklearn Decision Tree takes 0.8 seconds to train but produces a score of 0.659.
- Our Random Forest takes 156.4 seconds to train, for a score on the test set of 0.708.
- The sklearn Random Forest takes 1.7 seconds to train but produces a score of 0.709.
- Our Adaboost implementation run in 172,6 seconds, for a score on the test of 0.726.

## Conclusion :

Our first results are encouraging, we proved that the random forest is much more efficient than the decision tree, but we still have a long way to go on the optimization of the code and the method used to get closer to the execution speed of sklearn.
