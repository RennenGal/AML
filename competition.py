import os
import re
from pathlib import Path

import numpy as np


# Directory names. Do not change unless instructed by the TA.
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'competition_data'
OUTPUT_DIR = BASE_DIR / 'competition_output'


def train_predict(X_train, y_train, X_test):
    """Train a model and return predictions for X_test.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.

    Returns:
        np.ndarray: Predictions for X_test.
    """

    # --- Start of your code ---

    # Example: replace this with your model, preprocessing, tuning, etc.
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(max_iter=1000, random_state=42)
    clf = ...
    clf.fit(...)
    predictions = clf.predict(...)

    # --- End of your code ---

    return predictions


if __name__ == '__main__':
    # Enter the LAST 4 DIGITS of ONE group member's student ID, for example '0440'.
    # Submit one file per group under this single ID.
    student_id = '1385'  # CHANGE THIS TO THE LAST 4 DIGITS OF ONE GROUP MEMBER'S ID

    if student_id == 'XXXX':
        print("Error: change student_id from 'XXXX' to the last 4 digits of one group member's ID.")
        exit()
    if not re.fullmatch(r'\d{4}', student_id):
        print('Error: student_id must be exactly 4 digits (the last 4 digits of one group member ID).')
        exit()

    train_file = os.path.join(DATASETS_DIR, 'competition_train.npz')
    if not os.path.exists(train_file):
        print(f'Error: competition data file not found at {train_file}')
        print("Make sure the 'competition_data' folder is next to this script.")
        exit()

    data = np.load(train_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']

    print(f'Submission ID (group member): {student_id}')
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape:  {X_test.shape}')

    test_predictions = train_predict(X_train, y_train, X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred_file = os.path.join(OUTPUT_DIR, f'{student_id}_competition_predictions.npz')
    np.savez(pred_file, test_predictions=test_predictions)

    print(f'\nPredictions saved to {pred_file}')
    print(f'Submit ONLY the file: {student_id}_competition_predictions.npz')
