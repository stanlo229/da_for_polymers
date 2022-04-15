from sklearn import linear_model
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
from opv_ml.ML_models.sklearn.data.data import Dataset


FRAG_MASTER_DATA = pkg_resources.resource_filename(
    "opv_ml", "data/postprocess/frag_master_opv_ml_from_min.csv"
)

DATA_DIR = pkg_resources.resource_filename(
    "opv_ml", "data/process/master_opv_ml_from_min.csv"
)

dataset = Dataset(FRAG_MASTER_DATA, 0)
dataset.setup_frag()
# dataset = Dataset(DATA_DIR, 0)
# dataset.prepare_data()
# dataset.setup()

x_train = dataset.train_df["tokenized_input"].to_numpy()
y_train = dataset.train_df["PCE"].to_numpy()
x_train = x_train.tolist()
y_train = y_train.tolist()

x_test = dataset.test_df["tokenized_input"].to_numpy()
y_test = dataset.test_df["PCE"].to_numpy()
x_test = x_test.tolist()
y_test = y_test.tolist()

# ordinary least squares (ols)
model_ols = linear_model.LinearRegression(positive=True)

# non-negative least squares (nnls)
# reg_nnls = linear_model.LinearRegression(positive=True)
# model_ols = reg_nnls

# Ridge Regression
# reg = linear_model.Ridge(alpha=0.5)
# model_ols = reg

# RidgeCV
# reg = linear_model.RidgeCV(cv=10)
# model_ols = reg

# Lasso
# reg = linear_model.Lasso(alpha=0.1)
# model_ols = reg

model_ols.fit(x_train, y_train)
coef = model_ols.coef_
intercept = model_ols.intercept_

# Make predictions using the testing set
y_pred = model_ols.predict(x_test)

# plot outputs
m, b = np.polyfit(y_test, y_pred, 1,)
corr_coef = np.corrcoef(y_test, y_pred,)[0, 1]
fig, ax = plt.subplots()
ax.set_title("Predicted vs. Experimental PCE (%)")
ax.set_xlabel("Experimental_PCE_(%)")
ax.set_ylabel("Predicted_PCE_(%)")
ax.scatter(
    y_test, y_pred, s=4, alpha=0.7, color="#0AB68B",
)
# convert list back to array
y_test = np.array(y_test)
ax.plot(
    y_test, (m * y_test) + b, color="black", label=["r: " + str(corr_coef)],
)
ax.legend(loc="upper left")
textstr = "slope: " + str(m) + "\n" + "y-int: " + str(b)
ax.text(0.5, 0.5, textstr, wrap=True, verticalalignment="top")

plt.show()
