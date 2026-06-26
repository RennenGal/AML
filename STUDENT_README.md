# Competition: Student Instructions

## Goal

Build the best multiclass classifier for one shared competition dataset. Train on
`(X_train, y_train)`, predict labels for `X_test`, and submit one predictions
file per group.

## Setup

```bash
cd competition
python -m venv competition_env
source competition_env/bin/activate   # macOS/Linux
# competition_env\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Data

The `competition_data/competition_train.npz` file contains:

- `X_train` -- training features
- `y_train` -- training labels
- `X_test` -- test features to predict

The hidden test labels are not included.

## Steps

1. Open `competition.py`.
2. Set `student_id = 'XXXX'` to the **last 4 digits of ONE group member's student ID**
   (for example `'0440'`). Your group submits one file under this single ID.
3. Implement your model in `train_predict()`.
4. Run:

   ```bash
   python competition.py
   ```

5. Submit only the generated file:

   ```text
   competition_output/{student_id}_competition_predictions.npz
   ```

## Submission Rules

- Submit exactly one file per group, under the last 4 digits of one member's ID.
- The file name must be `{student_id}_competition_predictions.npz` (4 digits).
- The `.npz` file must contain the key `test_predictions`.
- The predictions must have exactly one label per row in `X_test`.
- The score earned by this submission is applied to every member of the group.

## Grading

Groups are ranked by hidden-test balanced accuracy. Rank 1 is best. For `n`
valid ranked groups, the competition score is:

```text
score = max(60, 100 * (1 - (rank - 1) / (n - 1)))
```
