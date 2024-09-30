# import sys
# # hacky import
# sys.path.insert(0, "../../Models/Base_Models")
# from GaussianNaiveBayes import GaussianNaiveBayes
# # weird import structure, open to refactoring and updating the proj structure
# # from ...Models.Base_Models.GaussianNaiveBayes import GaussianNaiveBayes

from sklearn.naive_bayes import GaussianNB
from load_data import load_train_data
from sklearn.metrics import  accuracy_score
if __name__ == '__main__':
    # load data
    x, y = load_train_data()
    gnb_clf = GaussianNB()
    gnb_clf.fit(x,y)
    print(gnb_clf.score(x,y))
    # print(accuracy_score(y,gnb_clf.predict(x)))

