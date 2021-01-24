from sklearn import tree
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def treeCreate():
    # Save original data
    titanic = pd.read_csv("titanikData.csv")

    # Preparing and saving data test
    titanic_test_header = pd.DataFrame(columns=['pclass','age','gender'])
    titanic_rows= pd.read_csv("titanikTest.csv")
    titanic_test = pd.concat([titanic_test_header, titanic_rows])
    titanic_test.columns.insert(0,['pclass','age','gender'])

    # convert string data into integer
    titanic_input = titanic
    titanic_input.gender = titanic.gender.map({'male': 1, 'female': 0})
    titanic_input.age = titanic.age.map({'adult': 1, 'child': 0})
    titanic_input.pclass = titanic.pclass.map({'crew': 0, '1st': 1, '2nd': 2, '3rd': 3})
    titanic_input.survived = titanic.survived.map({'yes': 1, 'no': 0})

    titanic_test['pclass'] = titanic_test['1st'].map({'crew': 0, '1st': 1, '2nd': 2, '3rd': 3})
    titanic_test['age'] = titanic_test.adult.map({'adult': 1, 'child': 0})
    titanic_test['gender'] = titanic_test.male.map({'male': 1, 'female': 0})
    titanic_test.drop(['1st', 'adult', 'male', 'yes'], axis=1, inplace=True)

    # save unique inputs
    titanic_unique_input = titanic_input.drop_duplicates()

    # for i in titanic_unique_input.values:
    #     print(i)

    # Preparing the decision tree for all the data
    # clf = tree.DecisionTreeClassifier()
    # clf.max_depth = 2
    # clf.criterion = 'entropy'
    # titanic_input_unique = titanic_input.drop_duplicates()
    # clf = clf.fit(titanic_input, titanic_target)
    # test = pd.DataFrame()
    # for i in titanic_input_unique.values:
    #     new_row = pd.Series({f'{str(([i[0], i[1], i[2]]))}'}: clf.predict([[i[0], i[1], i[2]]]))
    #     print(new_row)
    #     # print(([i[0], i[1], i[2]]), clf.predict([[i[0], i[1], i[2]]]))
    #     # test[str(([i[0], i[1], i[2]]))] = clf.predict([[i[0], i[1], i[2]]])
    #     # print([i[0], i[1], i[2]], clf.predict([[i[0], i[1], i[2]]]))
    # # print(test.unstack().head())

    predictions = list()

    # Create 100 dataset of size n. 65% are without replacement the other 35% with replacement
    for i in range(100):
        titanic_list_without_replacement = titanic_input.sample(frac=.632)
        titanic_size = len(titanic)
        titanic_list_without_replacement_size = len(titanic_list_without_replacement)
        complement_list_size = titanic_size - titanic_list_without_replacement_size
        titanic_list_with_replacement = titanic_list_without_replacement.sample(n=complement_list_size, replace=True)

        t_list = [titanic_list_without_replacement, titanic_list_with_replacement]
        titanic_list = pd.concat(t_list)
        titanic_target = titanic_list['survived']
        titanic_list = titanic_list.drop('survived', axis=1)

        clf = tree.DecisionTreeClassifier()
        clf.max_depth = 2
        clf.criterion = 'entropy'
        clf = clf.fit(titanic_list, titanic_target)
        prediction = clf.predict(titanic_test)
        print(prediction)
        # predictions.append(prediction)

    # array_of_predictions = np.array(predictions)

    # print(array_of_predictions)
    # print(predictions)
        # pred_matrix = np.array(predictions)
        # print(pred_matrix)


        # for ii in titanic_unique_input.values:
        #     for i in ii:
        #         print(i)
    #         rows.append([ii, clf.predict([ii])])
    #         # predictions[i] = [ii, clf.predict([ii])]
    #         # print(ii, clf.predict([ii]))
    #     predictions[headers[0]] = rows[0][0][0]
    #     predictions[headers[1]] = rows[0][0][1]
    #     predictions[headers[2]] = rows[0][0][2]
    # predictions.to_csv('test.csv')
        # print(predictions.unstack())
        # predicted = pd.concat(test.columns[0], clf.predict([[1,1,1]]))
        # print(predicted)
        # print(len(titanic_input_unique))
        # for ii in titanic_input_unique.values:
        #     print([ii[0], ii[1], ii[2]], clf.predict([[ii[0], ii[1], ii[2]]]))

        # accuracy = cross_val_score(clf, titanic_list, titanic_target, scoring='accuracy', cv=2)
        # print("Average Accuracy of  DT with depth ", clf.max_depth, " is: ", round(accuracy.mean(), 3))
        # # test.append(accuracy.mean())
        # precision = cross_val_score(clf, titanic_list, titanic_target, scoring='precision_weighted', cv=2)
        # print("Average precision_weighted of  DT with depth ", clf.max_depth, " is: ", round(precision.mean(), 3))


    # # import some data to play with
    # titanic = pd.read_csv("titanikData.csv", header=None)
    # # print(titanic)
    # iris = datasets.load_iris()
    #
    #
    # mylist = []
    # # do loop
    # clf = tree.DecisionTreeClassifier()
    # clf.max_depth = 10
    # clf.criterion = 'entropy'
    # clf = clf.fit(iris.data, iris.target)
    # # print("Decision Tree: ")
    # accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10)
    # # print("Average Accuracy of  DT with depth ", clf.max_depth, " is: ", round(accuracy.mean(), 3))
    # mylist.append(accuracy.mean())
    # precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted', cv=10)
    # print("Average precision_weighted of  DT with depth ", clf.max_depth, " is: ", round(precision.mean(), 3))
    # X = range(22)
    # plt.plot(X, [x * x for x in X])
    # plt.xlabel("This is the X axis")
    # plt.ylabel("This is the Y axis")
    # plt.show()
