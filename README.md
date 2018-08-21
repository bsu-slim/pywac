### Words-as-Classifiers in python

You can import as follows:

`from wac import WAC`

then you can train and evaluate a model. See example in `wac_test.py`

By default, the composition method is `prod` (`avg` and `sum` also available)

By default, the WAC classifier is LogisticRegression, but it can use any
type in the `classifier_spec`=(classifier,arguments) as long as the classifier
has `predict_proba`. 


