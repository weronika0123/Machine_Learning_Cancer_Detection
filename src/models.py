from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)

class KerasSigmoidBinaryWrapper:
    #Adapter: Keras model (sigmoid output) => sklearn-compatible interface (predict, predict_proba)
    def __init__(self, keras_model):
        self.model = keras_model
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        #Keras with sigmoid output returns shape (n, 1) or (n,)
        p1 = self.model.predict(X, verbose=0).ravel() #P(y=1)
        return np.column_stack([1.0 - p1, p1]) #[P(y=0), P(y=1)]
    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
    def get_keras_model(self):
        return self.model


def train_model(model_kind, model_params, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    use_val = X_val is not None and y_val is not None

    if model_kind == "Decision Tree":    
        dt_defaults = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",      
            "random_state": RANDOM_STATE
        }
        dt_defaults.update(model_params)
        model = DecisionTreeClassifier(**dt_defaults)

    elif model_kind == "Logistic Regression":
        max_iter = model_params.get("max_iter", 100)
        solver = model_params.get("solver", "lbfgs")
        penalty = model_params.get("penalty", "l2")
        C = model_params.get("C", 1.0)
        model = LogisticRegression(
            max_iter=max_iter, solver=solver, penalty=penalty, C=C, 
            class_weight="balanced", random_state=RANDOM_STATE
        )

    elif model_kind == "SVM":
        svm_defaults = {
            "kernel": "linear",                #'linear' | 'rbf' | 'poly' | 'sigmoid'
            "C": 1.0,
            "gamma": "scale",                  #for rbf/poly/sigmoid kernels
            "degree": 3,                       #for polynomial kernel
            "class_weight": "balanced",        #always applied for class imbalance
            "use_calibrated": True,            #for linear case: LinearSVC + CalibratedClassifierCV
            "calibration_method": "sigmoid",   #'sigmoid' | 'isotonic'
            "cv_calibration": 5,
            "probability": True                #required for SVC to enable predict_proba()
        }
        svm_defaults.update(model_params)
        kernel = svm_defaults["kernel"]
        if kernel == "linear" and svm_defaults["use_calibrated"]:
            #Efficient approach for high-dimensional data: LinearSVC + calibration
            base = LinearSVC(
                C=svm_defaults["C"],
                class_weight=svm_defaults["class_weight"],
                #dual=True (default) - suitable for many features and fewer samples
                #random_state not strictly required
            )
            model = CalibratedClassifierCV(
                estimator=base,
                method=svm_defaults["calibration_method"],
                cv=svm_defaults["cv_calibration"],
            )
            model_kind = "SVM linear calibrated"
        else:
            #Full SVC with probability=True (slower for large N or many features)
            model = SVC(
                kernel=kernel,
                C=svm_defaults["C"],
                gamma=svm_defaults["gamma"],
                degree=svm_defaults["degree"],
                class_weight=svm_defaults["class_weight"],
                probability=True, #required for ROC/PR analysis and threshold tuning
            )
        if (kernel == "linear" and model_kind != "SVM linear calibrated"):
            model_kind = "SVM linear"
    elif model_kind == "Deep Neural Network":
        dnn_defaults = {
            "hidden_layers": [128, 64],
            "activation": "relu",
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "class_weight": "balanced"
        }
        dnn_defaults.update(model_params)

        #Calculate class weights
        if dnn_defaults["class_weight"] == "balanced":
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train
            )
            class_weight_arg = {cls: w for cls, w in zip(classes, weights)}
        else:
            class_weight_arg = None

        #Neural network architecture
        model_keras = Sequential()
        model_keras.add(Dense(
            dnn_defaults["hidden_layers"][0],
            activation=dnn_defaults["activation"],
            input_shape=(X_train.shape[1],)
        ))
        model_keras.add(Dropout(dnn_defaults["dropout_rate"]))
        for units in dnn_defaults["hidden_layers"][1:]:
            model_keras.add(Dense(units, activation=dnn_defaults["activation"]))
            model_keras.add(Dropout(dnn_defaults["dropout_rate"]))
        model_keras.add(Dense(1, activation="sigmoid"))

        #Optimization setup
        optimizer = Adam(learning_rate=dnn_defaults["learning_rate"])
        model_keras.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        callbacks = []

        #Model training
        if use_val:
            model_keras.fit(
                X_train, y_train,
                epochs=dnn_defaults["epochs"],
                batch_size=dnn_defaults["batch_size"],
                validation_data=(X_val, y_val),
                verbose=1,
                callbacks=callbacks,
                class_weight=class_weight_arg
            )
        else:
            model_keras.fit(
                X_train, y_train,
                epochs=dnn_defaults["epochs"],
                batch_size=dnn_defaults["batch_size"],
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=callbacks,
                class_weight=class_weight_arg
            )
        #Wrap Keras model with sklearn-compatible interface (predict + predict_proba)
        model = KerasSigmoidBinaryWrapper(model_keras)

    elif model_kind == "XGBoost":
        xgb_defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "min_child_weight": 1,
            "scale_pos_weight": None, #will be calculated if None
            "random_state": RANDOM_STATE,
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        }
        xgb_defaults.update(model_params)
        
        #Calculate scale_pos_weight for class imbalance if not provided
        if xgb_defaults["scale_pos_weight"] is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            xgb_defaults["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1.0
            print(f"[XGBoost] Calculated scale_pos_weight: {xgb_defaults['scale_pos_weight']:.3f} (neg/pos = {neg_count}/{pos_count})")
        
        model = XGBClassifier(**xgb_defaults)
    else:
        raise ValueError(
            "Unknown model. Use one of: DecisionTree, LogisticRegression, SVM/SVC, DNN, or XGBoost."
        )
    print("Used X_train shape:", X_train.shape)
    print("Used X_test shape:", X_test.shape)
    if model_kind != "Deep Neural Network":
        model.fit(X_train, y_train)
    return (model, model_kind)
