from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV



RANDOM_STATE = 42

def train_model(model_kind, model_params, X_train, y_train, X_test, y_test):

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

    else:
        raise ValueError("Nieznany model. Użyj: DecisionTree/DT lub LogisticRegression/LR lub SVM/SVC")

    print("Used X_train shape:", X_train.shape)
    print("Used X_test shape:", X_test.shape)
    
    model.fit(X_train, y_train)
    return (model, model_kind)
    