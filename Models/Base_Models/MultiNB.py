import sklearn
class MultiNB(sklearn.naive_bayes.MultinomialNB):
    def __init__(self, alpha, fit_prior: bool, class_prior: [] = None):
        super().__init__(self)

