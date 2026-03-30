"""
train.py:
Entry point for the full training pipeline.
Run this once before launching the Streamlit app:

    python train.py- city Delhi

What it does:
  1. Runs preprocessing pipeline (load → clean → engineer → normalize → window)
  2. Builds the BiLSTM model
  3. Trains with callbacks
  4. Evaluates on test set → prints MAE + RMSE
  5. Saves model and scalers to outputs/
"""

import argparse
from src.model.architecture    import build_bilstm_model
from src.model.trainer         import train_model
from src.model.evaluator       import evaluate_model
from src.pipeline.preprocessor import run_preprocessing_pipeline

def main(city: str):
    print(f"\n{'='*60}")
    print(f"  AirGuard — Training Pipeline | City: {city}")
    print(f"{'='*60}\n")

    #Step 1:Preprocessing
    X_train, X_test, y_train, y_test, feat_scaler, target_scaler, feat_cols = \
        run_preprocessing_pipeline(city)

    #Step 2: Building model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (WINDOW_SIZE, n_features)
    model = build_bilstm_model(input_shape)

    #Step 3: Training
    history, model = train_model(model, X_train, y_train)

    #Step 4: Evaluating
    metrics, _ = evaluate_model(model, X_test, y_test, target_scaler)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  MAE  : {metrics['MAE']}")
    print(f"  RMSE : {metrics['RMSE']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AirGuard BiLSTM")
    parser.add_argument("--city", type=str, default="Delhi", help="City name from dataset")
    args = parser.parse_args()
    main(args.city)
