from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from util.plotka import plot_decision_regions


def main():
    dataset = make_moons(n_samples=500, noise=0.2)
    x = dataset[0]
    y = dataset[1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=1)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
    logistic_regression = LogisticRegression()
    svm = SVC(random_state=1)
    voting_classifier = VotingClassifier(estimators=[('rf', random_forest), ('lr', logistic_regression)])

    decision_tree.fit(x_train, y_train)
    random_forest.fit(x_train, y_train)
    logistic_regression.fit(x_train, y_train)
    svm.fit(x_train, y_train)
    voting_classifier.fit(x_train, y_train)

    plot_decision_regions(x, y, classifier=voting_classifier)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
