from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf



RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)

class KerasSigmoidBinaryWrapper:
    #Adapter: Keras (wyjście sigmoid) -> interfejs sklearn (predict, predict_proba)
    def __init__(self, keras_model):
        self.model = keras_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Keras z wyjściem sigmoid zwraca kształt (n, 1) albo (n,)
        p1 = self.model.predict(X, verbose=0).ravel()  # P(y=1)
        return np.column_stack([1.0 - p1, p1])         # [P(y=0), P(y=1)]

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
            max_iter=max_iter, solver=solver, penalty=penalty, C=C, class_weight="balanced", random_state=RANDOM_STATE
        )

    elif model_kind == "SVM":
    
        svm_defaults = {
        "kernel": "linear",          # 'linear' | 'rbf' | 'poly' | 'sigmoid'
        "C": 1.0,
        "gamma": "scale",            # dla rbf/poly/sigmoid
        "degree": 3,                 # dla poly
        "class_weight": "balanced",  #zawsze
        "use_calibrated": True,      # tylko dla linear: LinearSVC + CalibratedClassifierCV
        "calibration_method": "sigmoid",  # 'sigmoid' | 'isotonic'
        "cv_calibration": 5,
        "probability": True          # dla SVC (nieliniowe kernele) – żeby mieć predict_proba
    }
        svm_defaults.update(model_params)

        kernel = svm_defaults["kernel"]

        if kernel == "linear" and svm_defaults["use_calibrated"]:
            # Szybka wersja do wysokowymiarowych danych: LinearSVC + kalibracja
            base = LinearSVC(
                C=svm_defaults["C"],
                class_weight=svm_defaults["class_weight"],
                # dual=True (domyślnie) – w wielu cechach i mniejszej liczbie próbek jest OK
                # random_state nie jest wymagany
            )
            model = CalibratedClassifierCV(
                estimator=base,
                method=svm_defaults["calibration_method"],
                cv=svm_defaults["cv_calibration"],
            )
            model_kind="SVM linear calibrated"
        else:
            # Pełny SVC z proba=True (wolniejszy przy tysiącach cech i dużym N)
            model = SVC(
                kernel=kernel,
                C=svm_defaults["C"],
                gamma=svm_defaults["gamma"],
                degree=svm_defaults["degree"],
                class_weight=svm_defaults["class_weight"],
                probability=True,          # konieczne dla ROC/PR i tuningu progu
            )
        if (kernel == "linear" and model_kind!="SVM linear calibrated"):
            model_kind="SVM linear"

    elif model_kind == "DNN":
        dnn_defaults = {
            "hidden_layers": [128, 64],
            "activation": "relu",
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32
        }
        dnn_defaults.update(model_params)

        # Budowa sieci
        model_keras = Sequential()
        model_keras.add(Dense(dnn_defaults["hidden_layers"][0],
                              activation=dnn_defaults["activation"],
                              input_shape=(X_train.shape[1],)))
        model_keras.add(Dropout(dnn_defaults["dropout_rate"]))
        for units in dnn_defaults["hidden_layers"][1:]:
            model_keras.add(Dense(units, activation=dnn_defaults["activation"]))
            model_keras.add(Dropout(dnn_defaults["dropout_rate"]))
        model_keras.add(Dense(1, activation="sigmoid"))


        optimizer = Adam(learning_rate=dnn_defaults["learning_rate"])
        model_keras.compile(optimizer=optimizer,
                            loss="binary_crossentropy",
                            metrics=["accuracy"])


        # (opcjonalnie) prosty EarlyStopping – możesz dodać jeśli chcesz
        callbacks = []

        # Trening
        if use_val:
            model_keras.fit(
                X_train, y_train,
                epochs=dnn_defaults["epochs"],
                batch_size=dnn_defaults["batch_size"],
                validation_data=(X_val, y_val),
                verbose=1,
                callbacks=callbacks
            )
        else:
            model_keras.fit(
                X_train, y_train,
                epochs=dnn_defaults["epochs"],
                batch_size=dnn_defaults["batch_size"],
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=callbacks
            )


        # Owijamy w adapter z API sklearn (predict + predict_proba)
        model = KerasSigmoidBinaryWrapper(model_keras)

    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT, LogisticRegression/LR, SVM/SVC, lub Deep Neural Network/DNN")

    print("Used X_train shape:", X_train.shape)
    print("Used X_test shape:", X_test.shape)

    if model_kind != "DNN":
        model.fit(X_train, y_train)
    return (model, model_kind)
