import os
import logging
from datetime import datetime

from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


FEATURE_PATH = "path/to/feature.npz"

# Variables for logs
CURRENT_DIR = os.path.dirname(__file__)
LOG_BASE_DIR = os.path.join(CURRENT_DIR, 'logs')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_BASE_DIR, f'cumul_{TIMESTAMP}.txt')


def setup_logging():
    if not os.path.exists(LOG_BASE_DIR):
        os.makedirs(LOG_BASE_DIR)
    
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def load_data(feature_path):
    print("loading data...")
    x, y = load_svmlight_file(feature_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.toarray())
    x_test = scaler.transform(x_test.toarray())

    return x_train, x_test, y_train, y_test


def run_svm(x_train, x_test, y_train, y_test):
    print("running svm...")
    svm_model = svm.SVC(kernel='rbf')

    param_grid = {
        'C': [2**i for i in range(11, 18)],     # 2^11, ... ,2^17
        'gamma': [2**i for i in range(-3, 4)]   # 2^-3, ... ,2^3
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=30)

    # IMPORTANT: change n_jobs with enough CPU number you want to use
    grid_search = GridSearchCV(svm_model, param_grid, cv=kf, scoring='accuracy', verbose=3, n_jobs=64)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"feature: {FEATURE_PATH}")
    print(f"hyperparamers gird: {param_grid}")
    print(f"best score: {grid_search.best_score_}")
    print(f"best hyperparamers: {grid_search.best_params_}")
    print(f"best model's accuracy: {accuracy}")

    logging.info(f"feature: {FEATURE_PATH}")
    logging.info(f"hyperparamers gird: {param_grid}")
    logging.info(f"best score: {grid_search.best_score_}")
    logging.info(f"best hyperparamers: {grid_search.best_params_}")
    logging.info(f"best model's accuracy: {accuracy}")


if __name__ == "__main__":
    setup_logging()
    x_train, x_test, y_train, y_test = load_data(FEATURE_PATH)
    run_svm(x_train, x_test, y_train, y_test)