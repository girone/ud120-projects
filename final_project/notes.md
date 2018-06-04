# Notes on the project

## Preparation

### Available features (as described at the end of the PDF)

* Payments
  * Director fees
    Keiner der POIs hat hier einen nicht-leeren Eintrag.
  * Insgesammt ist der relative Anteil jeder payment-Kategorie am jeweiligen total payment wahrscheinlich aussagekräftiger, als absolute Werte.
* Stock value
  * Viele POIs haben einen hohen total stock value
* Email features
    I doubt that the email features (from_this_person_to_poi, from_poi_to_this_person) are admissible as given. This is because when splitting the data into training and test set, we assume the labels of the test set are unknown. However, the feature counts emails to all pois, not only those in the training set.

### Feature selection

Check individual features on their importance as done in [this example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html). I guess it is unlikely that there is a clean winner, but never mind trying.

### Ideas for new features

1. Emails from and to POIs as shares.
2. Set every payment and stock feature in relation to the total (maybe this can be achieved with PCA without adding N new features).
3. Text features. Think about some more complpex metric here.
4. Ratio of total payments to total stock

### Thoughts on algorithms

We have labelled data, with discrete labels. Thus, use supervised learning (Decision tree, naive Bayes, SVM, ensembles, k-nearest neighbors, LDA, logistic regression).

## Experiments

### 2018-05-31

First promising results: LinearSVC + feature "shared_receipt_with_poi" and the two new email features. Compared to LinearSVC + all features this is similar at first glance.

#### Next steps (1)

1. Look into the features, visualize them.
2. If there are outliers, think about how to remove them. Only if the results don't get much better think about new features.
3. Feature scaling.

### 2018-06-01

Wrote simple code to visualize features including annotation. This immediately showed that the entry "TOTAL" is an outlier in the dataset. Removed it from the data. Repeated the parameter sets from yesterday, they showed slightly better results.

Visualized the remaining data by plotting each variable against the others. Findings:

* deferred_income seems to have negative range
* loan advances has one high outlier
* salary and director_fees are mutally exclusive (either one is positive or the other)
  * POIs dont have director_fees
* total_payments has one outlier

Redo it with POIs highlighted. Include new features.

Some new results now: NaiveBayes outperforms LinearSVC.

GaussianNB(default parameters)
    Accuracy: 0.74713    Precision: 0.23578    Recall: 0.40000    F1: 0.29668    F2: 0.35109
    Total predictions: 15000    True positives:  800    False positives: 2593    False negatives: 1200    True negatives: 10407

LinearSVC(default parameters)
    Accuracy: 0.74087    Precision: 0.21208    Recall: 0.34750    F1: 0.26341    F2: 0.30815
    Total predictions: 15000    True positives:  695    False positives: 2582    False negatives: 1305    True negatives: 10418

#### Next steps (2)

1. Remove outliers systematically.
2. Do some fine-tuning, some scaling.
3. Extract text features: Top words by phrases, create clusters, see if there are more frequent terms for POIs.

### 2018-06-02 (1)

Read about outlier detection with sklearn. LOF seems to be fitting for the present task and data set. Implemented it, using the featureFormat function given used by the tester code. Had problems with a different array lenght of the outlier labels and the keys in the data. Solved it by using `featureFormat()` in a way that NaN is translated to 0.0, but entries with all 0.0 values are kept.

I use all available features (not the two computed email features) for the outlier detection. For the results I compare Naive Bayes and SVC with the default parameters. _Note_ that the nearest neighbor distance works best with equidistant dimensions, which might not be true for all features. Some feature scaling could improve the results. However, I feel for now this is good enough.

#### 10% outliers

Changed results:

GaussianNB(default parameters)
    Accuracy: 0.37792    Precision: 0.09719    Recall: 0.85500    F1: 0.17454    F2: 0.33406
    Total predictions: 13000    True positives:  855    False positives: 7942    False negatives:  145    True negatives: 4058

LinearSVC(default parameters)
    Accuracy: 0.73662    Precision: 0.12149    Recall: 0.38900    F1: 0.18515    F2: 0.27006
    Total predictions: 13000    True positives:  389    False positives: 2813    False negatives:  611    True negatives: 9187

Gaussian Naive Bayes has worse accuracy, poor precision but good recall now. Maybe too many outliers have been removed (5 out of 15 outliers are POIs).

#### 5% outliers

Removing the 5% outliers found this way, we remove 8 outliers (3 of which are POIs, which represents an even larger share). The Naive Bayes's precision gets better, and SVCs precision also looks promising now, while recall does not change much. Time to work on the algorithm parameters.

GaussianNB(default parameters)
    Accuracy: 0.33579    Precision: 0.14979    Recall: 0.78050    F1: 0.25135    F2: 0.42370
    Total predictions: 14000    True positives: 1561    False positives: 8860    False negatives:  439    True negatives: 3140

LinearSVC(default parameters)
    Accuracy: 0.72829    Precision: 0.22667    Recall: 0.37400    F1: 0.28226    F2: 0.33097
    Total predictions: 14000    True positives:  748    False positives: 2552    False negatives: 1252    True negatives: 9448

#### Next steps (3)

1. Do feature scaling prior to outlier removal.
2. Try other algorithms and do some parameter tuning.
3. Try to extract the aforementioned email features.

### 2018-06-02 (2)

Use `scale()` and `robust_scale()` to scale features. Might improve results of SVC, not Naive Bayes.

Results got quite a lot better, also less POIs have been removed as outliers. Thus, also for Naive Bayes the results improved (by a large extend, actually). However, applying it seemed too easy, so I need to find a way to validate my code.

Added argparse to the `poi_id.py` script, to control the preprocessing and used algorithm without changing the code. Need to set the default to the values I choose for the submission when cleaning up.

One more change: Remove the `"TOTAL"` entry before feature scaling, to reduce its bias on the data. Indeed, this does not change the results when using `robust_scale()`, but it seems fair to remove it before `scale()`.

Observation during validation: NaN values become numbers. They are probably treated as zeros in the input. This seems to be valid, because it's is just what the tester code does. The validation shows no problems so far. So here are the next results:

#### Preprocess with `scale()`

GaussianNB(default parameters)
    Accuracy: 0.34179    Precision: 0.14552    Recall: 0.74050    F1: 0.24325    F2: 0.40738
    Total predictions: 14000    True positives: 1481    False positives: 8696    False negatives:  519    True negatives: 3304

LinearSVC(default parameters)
    Accuracy: 0.83036    Precision: 0.33480    Recall: 0.19000    F1: 0.24242    F2: 0.20799
    Total predictions: 14000    True positives:  380    False positives:  755    False negatives: 1620    True negatives: 11245

#### Preprocess with `robust_scale()`

GaussianNB(default parameters)
    Accuracy: 0.31907    Precision: 0.17211    Recall: 0.98850    F1: 0.29317    F2: 0.50726
    Total predictions: 14000    True positives: 1977    False positives: 9510    False negatives:   23    True negatives: 2490

LinearSVC(default parameters)
    Accuracy: 0.80521    Precision: 0.28854    Recall: 0.24800    F1: 0.26674    F2: 0.25517
    Total predictions: 14000    True positives:  496    False positives: 1223    False negatives: 1504    True negatives: 10777

Using the robust feature scaling seems to give little less accuracy and precision, but increases recall. Especially for Gaussian Naive Bayes there is a noteworthy improvement.

#### Next steps (4)

1. Think about where and how to apply PCA
2. Try other algorithms
3. Try different parameter sets
4. Check automatic feature selection
5. Advanced features from text

### 2018-06-03

PCA should be done using a pipeline. Maybe this can include the feature scaling and outlier removal, too. Pipelines can be created using `make_pipeline()` and supplying any order of classifiers as arguments. Pipeline objects can be parameterized dynamically.

First try for using PCA resulted in unchanged prediction quality for Gaussian Naive Bayes, and much worse performance for Linear SVM. But it consumes more time.

Read forums to get an idea if I was using PCA in a wrong way. Instead of hints, I found other peoples reporting about their use of [GridSearchCV for parameter tuning](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in combination with [cv=StratifiedShuffleSplit()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) and [RFECV for feature selection](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html). Btw. the CV stands for Cross Validation. Also read about [Gradient-boosted machines](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) which seem to be among the best performing algorithms. Will try those in my solution.

Added a new feature "ratio between total payments and stock value". This boosts performance of especially the SVM (whereas Naive Bayes does not change, not even a bit, so it is not reported again):

LinearSVC(default parameters)
    Accuracy: 0.81136    Precision: 0.32629    Recall: 0.30100    F1: 0.31313    F2: 0.30574
    Total predictions: 14000    True positives:  602    False positives: 1243    False negatives: 1398    True negatives: 10757

Yet again, adding PCA decreases the quality of the results by a factor of 2. Even when playing with the parameters (e.g. set n_components to half the number of features). So I will skip it for now.

Since using PCA in the tester code already hinted that computing times grows to a non-trivial amount, I filled in the code for evaluation within the `poi_id.py` code.s

Added parameters for `--feature-selection` which can be `None`, `SelectKBest`, `RFECV`. First version had some bug or wrong setting, because SVM would not assign _any_ person to the POIs. Need to understand the score which the feature selection outputs for each feature. Does the features with high or low score get selected.

Not sure yet if there is a Wechselwirkung between feature_selection and feature_scaling and if it is good or bad.

The validation code is somewhat unclear. Not sure if I messed it up or if it has been given. Check git history tomorrow!

#### Next steps (5)

1. Review  the last code. Could not bring it to a good state. Finish it.
2. Get feature selection right.

### 2018-06-04

Looking into SelectKBest feature selection, finally understood that the features with the highest score remain. However, the scores are really close to each other and I wonder if there is much difference from the present features.

When printing the three new features, I noticed that they should also be part of the outlier removal.

Played around with Recursive Feature Elimination Cross Validated (RFECV) and found that it gives varying results on each run. The resulting classifier does not label any person as POI and thus the evaluation cannot run. I need to clean up the code again and work more systematically on this.

Also tried SelectPercentile(percentile=68.5) for feature selection. The result is easier to understand and results are quite good for the two classifiers used so far:

Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5)), ('gaussiannb', GaussianNB(priors=None))])
    Accuracy: 0.75057    Precision: 0.25748    Recall: 0.39600    F1: 0.31206    F2: 0.35753
    Total predictions: 14000    True positives:  792    False positives: 2284    False negatives: 1208    True negatives: 9716

Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5)), ('linearsvc', LinearSVC(default_parameters))])
    Accuracy: 0.82107    Precision: 0.24826    Recall: 0.12450    F1: 0.16583    F2: 0.13829
    Total predictions: 14000    True positives:  249    False positives:  754    False negatives: 1751    True negatives: 11246

Results for LinearSVC really depend on the scaling. Seems like normal scaling gives best accuracy and precision but lowest recall, no scaling is on the other side of the scale and robust scaling somewhere between the two. Need to switch to a non-linear kernel anyhow.

Running experiments with `RFECV` against `tester.py` is a PITA. The evaluation in `poi_id.py` is very unstable. Need to change the latter so that it returns fast yet stable results.

Cleaned the code (remove feature selection experiments, make outlier removal part of the pipeline).

Use StratifiedShuffleSplit for more CV-ish evaluation in `poi_id.py` so that I get fast and stable results during development.

#### Next steps (6)

1. Set up GridSearchCV to help with finding optimal algorithm and parameter settings.
2. Add some more relative metrics for the financial data.
3. Find good algo+params combo.
4. PCA, new text features, ...