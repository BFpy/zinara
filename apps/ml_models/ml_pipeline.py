import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import shap
import os
from django.conf import settings
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

class VehicleLicensePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_type = None

    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the vehicle licensing data"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} records from {data_path}")

            # Data cleaning and preprocessing
            df = self._clean_data(df)
            df = self._engineer_features(df)

            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _clean_data(self, df):
        """Clean the dataset following best practices"""
        logger.info("Starting data cleaning process...")

        # 1. Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['vehicle_id'])
        logger.info(f"Removed {initial_count - len(df)} duplicate records")

        # 2. Handle missing values intelligently

        # For payment modes - if payment is in-person/cash, online platform data should be null
        # This is normal behavior, not missing data
        online_payment_mask = df['payment_mode'].isin(['online', 'mobile_money'])

        # Fill missing online platform data only for online payments
        if 'online_platform_last_login' in df.columns:
            df.loc[~online_payment_mask, 'online_platform_last_login'] = None
            df.loc[online_payment_mask & df['online_platform_last_login'].isna(), 'online_platform_last_login'] = df['last_license_renewal']

        # 3. Handle date inconsistencies
        date_columns = ['registration_date', 'last_license_renewal', 'license_expiry_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 4. Remove records with impossible dates (future registration dates, etc.)
        if 'registration_date' in df.columns:
            df = df[df['registration_date'] <= pd.Timestamp.now()]

        # 5. Handle agent sync delays - if payment is not through agent, sync delay should be 0
        if 'agent_sync_delay' in df.columns and 'payment_mode' in df.columns:
            agent_payment_mask = df['payment_mode'] == 'agent'
            df.loc[~agent_payment_mask, 'agent_sync_delay'] = 0

            # For agent payments, fill missing sync delays with median
            median_sync_delay = df.loc[agent_payment_mask, 'agent_sync_delay'].median()
            df.loc[agent_payment_mask & df['agent_sync_delay'].isna(), 'agent_sync_delay'] = median_sync_delay

        # 6. Clean vehicle types - standardize naming
        if 'vehicle_type' in df.columns:
            df['vehicle_type'] = df['vehicle_type'].str.lower().str.strip()
            # Standardize common variations
            type_mapping = {
                'car': 'sedan',
                'automobile': 'sedan',
                'motorbike': 'motorcycle',
                'bike': 'motorcycle',
                'lorry': 'truck',
                'pickup': 'truck'
            }
            df['vehicle_type'] = df['vehicle_type'].replace(type_mapping)

        # 7. Handle renewal counts - ensure consistency
        if 'total_renewals' in df.columns and 'late_renewals_count' in df.columns:
            # Late renewals cannot exceed total renewals
            df.loc[df['late_renewals_count'] > df['total_renewals'], 'late_renewals_count'] = df['total_renewals']

        # 8. Remove outliers using IQR method for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['vehicle_id', 'is_licensed']:  # Don't remove outliers from ID or target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                if outlier_count > 0:
                    logger.info(f"Removing {outlier_count} outliers from {col}")
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        logger.info(f"Data cleaning complete. Final dataset size: {len(df)}")
        return df

    def _engineer_features(self, df):
        """Engineer features for model training"""
        logger.info("Engineering features...")

        # 1. Days since last renewal
        if 'last_license_renewal' in df.columns:
            df['days_since_renewal'] = (pd.Timestamp.now() - df['last_license_renewal']).dt.days
            df['days_since_renewal'] = df['days_since_renewal'].fillna(365)  # Default to 1 year

        # 2. Vehicle age
        if 'registration_date' in df.columns:
            df['vehicle_age_years'] = (pd.Timestamp.now() - df['registration_date']).dt.days / 365.25

        # 3. Renewal frequency (renewals per year of vehicle life)
        if 'total_renewals' in df.columns and 'vehicle_age_years' in df.columns:
            df['renewal_frequency'] = df['total_renewals'] / (df['vehicle_age_years'] + 1)  # +1 to avoid division by zero

        # 4. Late renewal rate
        if 'late_renewals_count' in df.columns and 'total_renewals' in df.columns:
            df['late_renewal_rate'] = df['late_renewals_count'] / (df['total_renewals'] + 1)

        # 5. Season of last renewal
        if 'last_license_renewal' in df.columns:
            df['renewal_month'] = df['last_license_renewal'].dt.month
            df['renewal_quarter'] = df['last_license_renewal'].dt.quarter

        # 6. Payment digitalization score (higher for online payments)
        if 'payment_mode' in df.columns:
            digital_mapping = {
                'online': 1.0,
                'mobile_money': 0.8,
                'agent': 0.4,
                'cash': 0.0
            }
            df['payment_digital_score'] = df['payment_mode'].map(digital_mapping)

        # 7. Regional risk score (can be updated based on historical data)
        if 'region' in df.columns:
            region_risk = {
                'urban': 0.3,
                'peri_urban': 0.5,
                'rural': 0.7
            }
            df['region_risk_score'] = df['region'].map(region_risk)

        # 8. Agent efficiency (inverse of sync delay)
        if 'agent_sync_delay' in df.columns:
            max_delay = df['agent_sync_delay'].max()
            df['agent_efficiency'] = 1 - (df['agent_sync_delay'] / (max_delay + 1))

        logger.info("Feature engineering complete")
        return df

    def prepare_features_target(self, df, target_column='is_licensed'):
        """Prepare features and target for model training"""

        # Define feature columns
        feature_cols = [
            'days_since_renewal', 'vehicle_age_years', 'total_renewals',
            'late_renewals_count', 'average_renewal_delay', 'renewal_frequency',
            'late_renewal_rate', 'payment_digital_score', 'region_risk_score',
            'agent_efficiency', 'renewal_month', 'renewal_quarter'
        ]

        # Add categorical features with encoding
        categorical_features = []

        # Vehicle type encoding
        if 'vehicle_type' in df.columns:
            vehicle_type_encoded = pd.get_dummies(df['vehicle_type'], prefix='vehicle_type')
            df = pd.concat([df, vehicle_type_encoded], axis=1)
            categorical_features.extend(vehicle_type_encoded.columns.tolist())

        # Region encoding
        if 'region' in df.columns:
            region_encoded = pd.get_dummies(df['region'], prefix='region')
            df = pd.concat([df, region_encoded], axis=1)
            categorical_features.extend(region_encoded.columns.tolist())

        # Payment mode encoding
        if 'payment_mode' in df.columns:
            payment_encoded = pd.get_dummies(df['payment_mode'], prefix='payment')
            df = pd.concat([df, payment_encoded], axis=1)
            categorical_features.extend(payment_encoded.columns.tolist())

        # Combine all features
        all_features = feature_cols + categorical_features

        # Filter existing columns
        available_features = [col for col in all_features if col in df.columns]
        self.feature_columns = available_features

        # Prepare X and y
        X = df[available_features].fillna(0)
        y = ~df[target_column]  # Invert because we want to predict unlicensed (True for unlicensed)

        logger.info(f"Prepared {len(available_features)} features for training")
        logger.info(f"Target distribution - Unlicensed: {y.sum()}, Licensed: {(~y).sum()}")

        return X, y

    def train_model(self, X, y, model_type='xgboost', test_size=0.2, random_state=42):
        """Train the specified model"""
        logger.info(f"Training {model_type} model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train model based on type
        self.model_type = model_type

        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=random_state, class_weight='balanced')
            self.model.fit(X_train_scaled, y_train)

        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
            self.model.fit(X_train, y_train)  # Trees don't need scaling

        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=random_state, class_weight='balanced'
            )
            self.model.fit(X_train, y_train)

        elif model_type == 'xgboost':
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            self.model = xgb.XGBClassifier(
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            self.model.fit(X_train, y_train)

        # Evaluate model
        if model_type in ['logistic_regression']:
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info(f"Model training complete. Metrics: {metrics}")
        return metrics

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model doesn't support feature importance")

        feature_importance = dict(zip(self.feature_columns, importance_scores))

        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return sorted_features

    def save_model(self, model_path, model_name):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        os.makedirs(model_path, exist_ok=True)

        model_file = os.path.join(model_path, f"{model_name}.joblib")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.joblib")
        features_file = os.path.join(model_path, f"{model_name}_features.joblib")

        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_columns, features_file)

        logger.info(f"Model saved to {model_file}")
        return model_file

    def load_model(self, model_path, model_name):
        """Load a saved model"""
        model_file = os.path.join(model_path, f"{model_name}.joblib")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.joblib")
        features_file = os.path.join(model_path, f"{model_name}_features.joblib")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_columns = joblib.load(features_file)

        logger.info(f"Model loaded from {model_file}")

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not loaded")

        if self.model_type == 'logistic_regression':
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities