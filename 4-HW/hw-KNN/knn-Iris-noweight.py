import numpy as np

from sklearn import neighbors, datasets
# datasets: avai lib Iris
# neighbors: avai lib KNN

from sklearn.model_selection import train_test_split
# slit data into train + test sets

from sklearn.metrics import accuracy_score
# evaluate the accuracy of the model

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#print(iris.data)
#print(iris.feature_names)
#print(iris_X.shape)
#print(iris_y.shape)
#print("Labels:", np.unique(iris_y))

X_train, X_test, y_train, y_test = train_test_split(
    iris_X,
    iris_y,
    test_size=130
)
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Choose the KNN model at runtime.
k = int(input("Chọn model (...KNN): ").strip())

# myweight function
def myweight1(distances):
    return 1 / (distances + 1e-8)

def myweight2(distances):
    sigma2 = 0.4
    return np.exp(-(distances ** 2) / (sigma2))

# choose  weights for KNN model
weights = input("option :uniform, distance, myweight1, myweight2: ").strip()

model = neighbors.KNeighborsClassifier(
    n_neighbors=k,
    p=2,  # Euclidean distance, L2
    weights=weights  
    # uniform weights --> all neighbors have the same weight, default is uniform
)


# remember to fit the model with training data
model.fit(X_train, y_train)

# predict labels for test data
y_pred = model.predict(X_test)

# score the accuracy of the model
print(
    f"Accuracy of {k}NN: %.2f %%"
    % (100 * accuracy_score(y_test, y_pred))
)

