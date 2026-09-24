"""
Model Training Module
Train and evaluate regression models for salary prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
import json


class SalaryPredictor:
    """Trains and manages salary prediction models"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()

    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare"""

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\n" + "=" * 70)
        print("TRAINING MODELS")
        print("=" * 70)

        # ===== Model 1: Gradient Boosting =====
        print("\n1️⃣  Training Gradient Boosting...")

        gbm = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        gbm.fit(X_train_scaled, y_train)
        self.models["Gradient Boosting"] = gbm

        # ===== Model 2: XGBoost =====
        print("2️⃣  Training XGBoost...")

        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )

        xgb_model.fit(X_train_scaled, y_train)
        self.models["XGBoost"] = xgb_model

        # ===== Model 3: Random Forest =====
        print("3️⃣  Training Random Forest...")

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        rf_model.fit(X_train_scaled, y_train)
        self.models["Random Forest"] = rf_model

        # ===== Model 4: Ridge Regression =====
        print("4️⃣  Training Ridge Regression...")

        ridge = Ridge(alpha=1.0)

        ridge.fit(X_train_scaled, y_train)
        self.models["Ridge"] = ridge

        print("\n" + "=" * 70)
        print("EVALUATING MODELS")
        print("=" * 70)

        best_mae = float("inf")

        for name, model in self.models.items():

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Avoid division by zero in MAPE
            non_zero = y_test != 0

            if non_zero.any():
                mape = (
                    np.mean(
                        np.abs(
                            (y_test[non_zero] - y_pred[non_zero])
                            / y_test[non_zero]
                        )
                    )
                    * 100
                )
            else:
                mape = 0

            # Cross-validation
            cv_scores = cross_val_score(
                model,
                X_train_scaled,
                y_train,
                cv=5,
                scoring="r2"
            )

            self.results[name] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }

            print(f"\n{name}:")
            print(f"  MAE:  ₹{mae:.2f}")
            print(f"  RMSE: ₹{rmse:.2f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(
                f"  CV (R²): {cv_scores.mean():.4f} "
                f"± {cv_scores.std():.4f}"
            )

            if mae < best_mae:
                best_mae = mae
                self.best_model = model
                self.best_model_name = name

        print("\n" + "=" * 70)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print("=" * 70)

        return self.best_model, self.best_model_name

    def get_feature_importance(self, feature_names):
        """Get feature importance from best model"""

        if not hasattr(self.best_model, "feature_importances_"):
            print("⚠️ Model doesn't have feature_importances_")
            return None

        importance = self.best_model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values(
            "importance",
            ascending=False
        )

        return feature_importance_df

    def save_models(self, model_dir="models"):
        """Save all trained models"""

        os.makedirs(model_dir, exist_ok=True)

        # Save best model
        with open(f"{model_dir}/best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        # Save scaler
        with open(f"{model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save results
        results_clean = {
            k: {
                kk: float(vv)
                for kk, vv in v.items()
            }
            for k, v in self.results.items()
        }

        with open(
            f"{model_dir}/model_results.json",
            "w"
        ) as f:
            json.dump(
                results_clean,
                f,
                indent=2
            )

        print(f"\n✅ Models saved to {model_dir}/")
        print(f"   - best_model.pkl ({self.best_model_name})")
        print("   - scaler.pkl")
        print("   - model_results.json")


def train_models():
    """Main training pipeline"""

    print("📂 Loading processed data...")

    X = pd.read_csv(
        "data/processed/X_features.csv"
    )

    y = pd.read_csv(
        "data/processed/y_target.csv"
    ).iloc[:, 0]

    print(
        f"✅ Loaded features: {X.shape}, "
        f"target: {y.shape}"
    )

    # ==========================================================
    # CLEAN MISSING VALUES
    # ==========================================================

    print("\n🔍 Checking missing values...")

    x_nan = X.isna().sum().sum()
    y_nan = y.isna().sum()

    print(f"   X NaN values: {x_nan}")
    print(f"   y NaN values: {y_nan}")

    # Combine X and y so the same rows are removed from both
    data = X.copy()
    data["__target__"] = y.values

    before = len(data)

    data = data.dropna()

    removed = before - len(data)

    X = data.drop(columns=["__target__"])
    y = data["__target__"]

    print(f"✅ Removed {removed} rows containing NaN")
    print(
        f"✅ Clean data: X = {X.shape}, "
        f"y = {y.shape}"
    )

    # ==========================================================
    # SPLIT DATA
    # ==========================================================

    print("\n📊 Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(
        f"✅ Train: {X_train.shape}, "
        f"Test: {X_test.shape}"
    )

    # ==========================================================
    # TRAIN MODELS
    # ==========================================================

    predictor = SalaryPredictor()

    best_model, best_name = predictor.train_all_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    print("\n📊 Top 15 Important Features:")

    importance_df = predictor.get_feature_importance(
        X.columns
    )

    if importance_df is not None:
        print(importance_df.head(15))

    # ==========================================================
    # SAVE MODELS
    # ==========================================================

    predictor.save_models()

    print("\n✅ Training complete!")


if __name__ == "__main__":
    train_models()