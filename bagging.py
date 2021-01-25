from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)


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
    features = titanic.columns.drop('survived')

    for i in range(100):
        # Create 100 dataset of size n. 65% are without replacement the other 35% with replacement
        titanic_list_without_replacement = titanic_training_unique_values.sample(frac=.632)
        titanic_size = len(titanic)
        titanic_list_without_replacement_size = len(titanic_list_without_replacement)
        complement_list_size = titanic_size - titanic_list_without_replacement_size
        titanic_list_with_replacement = titanic_list_without_replacement.sample(n=complement_list_size, replace=True)

        titanic_list = pd.concat([titanic_list_without_replacement, titanic_list_with_replacement])
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
        # On prediction_decision the value for survived is 1 and 0 to not survived.
        # If sum is more than 50% so there is more 1 than 0 and by majority the person survived
        prediction_decision[i] = 'yes' if prediction_decision[i] > len(array_of_predictions)/2 else 'no'

    predict_decision = pd.DataFrame(prediction_decision, columns=['prediction'])
    titanic_test_data = pd.concat([titanic_test_data, predict_decision], axis=1)

    print(titanic_test_data)

    print("Successfully predicted data with accuracy " + "= "
          + str(accuracy_score(titanic_test_data.survived, titanic_test_data.prediction)*100) + ' %')


def main():
    bagging()


if __name__ == "__main__":
    main()
