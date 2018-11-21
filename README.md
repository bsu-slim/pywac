# Words-as-Classifiers (Python)


### Requirements

- scikit-learn
- pandas
- pandasql
- sqlite3
- numpy

You can import as follows:

`from wac import WAC`

then you can train and evaluate a model. See example in `wac_test.py`

By default, the composition method is `prod` (`avg` and `sum` also available)

By default, the WAC classifier is LogisticRegression, but it can use any
type in the `classifier_spec`=(classifier,arguments) as long as the classifier
has `predict_proba`. 

### Notes

Obviously, wac is a simple application of a classifier for each word and is therefore easy to implement. This code helps with taking care of all the vocabulary, training data, and evaluation. It's not the same code or dataset in the original paper (see below); rather, it's a generalized implemenetation. 


### Reference

```
@InProceedings{kennington-schlangen:2015:ACL-IJCNLP,
  author    = {Kennington, Casey  and  Schlangen, David},
  title     = {Simple Learning and Compositional Application of Perceptually Grounded Word Meanings for Incremental Reference Resolution},
  booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  month     = {July},
  year      = {2015},
  address   = {Beijing, China},
  publisher = {Association for Computational Linguistics},
  pages     = {292--301},
  url       = {http://www.aclweb.org/anthology/P15-1029}
}
```

