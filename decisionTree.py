from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def convert_columns_into_integers_values(dataframe):
    # Convert string dataframe values in integers
    new_dataframe = dataframe.copy()
    new_dataframe.pclass = dataframe.pclass.map({'crew': 0, '1st': 1, '2nd': 2, '3rd': 3})
    new_dataframe.gender = dataframe.gender.map({'male': 1, 'female': 0})
    new_dataframe.age = dataframe.age.map({'adult': 1, 'child': 0})
    new_dataframe.survived = dataframe.survived.map({'yes': 1, 'no': 0})
    return new_dataframe


def bagging():
    # Save original data into data frames
    titanic = pd.read_csv("titanikData.csv")
    header = titanic.columns.values
    titanic_test_data = pd.read_csv("titanikTest.csv", names=header)

    titanic_test = convert_columns_into_integers_values(titanic_test_data)
    titanic_training_unique_values = titanic.drop_duplicates()

    predictions = []
    titanic_test_input = titanic_test.drop('survived', axis=1)
    features = ['pclass', 'age', 'gender']

    # Create 100 dataset of size n. 65% are without replacement the other 35% with replacement
    for i in range(100):
        titanic_list_without_replacement = titanic_training_unique_values.sample(frac=.632)
        titanic_size = len(titanic)
        titanic_list_without_replacement_size = len(titanic_list_without_replacement)
        complement_list_size = titanic_size - titanic_list_without_replacement_size
        titanic_list_with_replacement = titanic_list_without_replacement.sample(n=complement_list_size, replace=True)

        t_list = [titanic_list_without_replacement, titanic_list_with_replacement]
        titanic_list = pd.concat(t_list)
        titanic_list = convert_columns_into_integers_values(titanic_list)

        x = titanic_list[features]
        y = titanic_list.survived

        clf = tree.DecisionTreeClassifier()
        clf.max_depth = 1
        clf.criterion = 'entropy'
        clf = clf.fit(x, y)

        prediction = clf.predict(titanic_test_input)
        predictions.append(prediction)

    array_of_predictions = np.array(predictions)
    prediction_decision = [sum(x) for x in zip(*array_of_predictions)]
    for i in range(len(prediction_decision)):
        prediction_decision[i] = 'yes' if prediction_decision[i] > 50 else 'no'
    predict_decision = pd.DataFrame(prediction_decision, columns=['prediction'])
    titanic_test_data = pd.concat([titanic_test_data, predict_decision], axis=1)
    print("accuracy " + "= " + str(accuracy_score(titanic_test_data.survived, titanic_test_data.prediction)))


def main():
    bagging()


if __name__ == "__main__":
    main()
