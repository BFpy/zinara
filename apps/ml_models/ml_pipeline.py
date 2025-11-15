import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    # Logger will be initialized later, so we'll log the warning when SMOTE is actually needed
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
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_columns = None
        self.all_feature_columns = None  # keep full feature list before selection
        self.feature_selector = None
        self.model_type = None
        self.threshold_ = 0.5
        self.use_smote = True  # Use SMOTE for class imbalance
        self.use_feature_selection = True  # Use feature selection
        self.categorical_categories = {}  # Store expected categories for consistent encoding

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map enriched CSV column names (Title Case) to internal snake_case expected here."""
        col_map = {
            'Vehicle ID': 'vehicle_id',
            'Owner ID': 'owner_id',
            'Make': 'make',
            'Model': 'model',
            'Year': 'year',
            'Vehicle Type': 'vehicle_type',
            'License Status': 'license_status',
            'Last Renewal Date': 'last_license_renewal',
            'Expiration Date': 'license_expiry_date',
            'Fine Amount': 'fine_amount',
            'Agent Service Used': 'agent_service_used',
            'Online Platform Used': 'online_platform_used',
            'Compliance History': 'compliance_history',
            'Geographic Location': 'geographic_location',
            'Region': 'region',
            'Income Level': 'income_level',
            'Number of Vehicles Owned': 'total_vehicles_owned',
            'Previous Violations': 'previous_violations',
            'Agent Hours Used': 'agent_hours_used',
            'User Feedback Score': 'user_feedback_score',
            'Predictive Score': 'predictive_score',
            'Preferred Payment Mode': 'payment_mode',
            'Date of Data Collection': 'data_collection_date',
            'Days Since Last Renewal': 'days_since_renewal',
            'Number of Late Renewals in Last 3 Years': 'late_renewals_count',
            'Average Renewal Lag Days': 'average_renewal_delay',
            'Agent Synchronization Lag': 'agent_sync_delay',
            'Month': 'renewal_month',
            'Quarter': 'renewal_quarter',
            'is_licensed': 'is_licensed',
            # Additional mappings for notebook features
            'Total Renewals': 'total_renewals',
            'Late Renewals Count': 'late_renewals_count',
            'Agent Sync Delay': 'agent_sync_delay',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure categorical columns are treated as strings to avoid Pandas Categorical setitem errors
        cat_cols = df.select_dtypes(include=['category']).columns.tolist()
        for col in cat_cols:
            # Convert to string, handling NaN values
            df[col] = df[col].astype(str).replace('nan', '')

        # Normalize common categorical values
        if 'vehicle_type' in df.columns:
            df['vehicle_type'] = df['vehicle_type'].astype(str).str.lower().str.strip()
        if 'region' in df.columns:
            df['region'] = df['region'].astype(str).str.lower().str.replace('-', '_', regex=False).str.strip()
        if 'payment_mode' in df.columns:
            df['payment_mode'] = df['payment_mode'].astype(str).str.lower().str.replace(' ', '_').str.strip()
        if 'license_status' in df.columns:
            df['license_status'] = df['license_status'].astype(str).str.lower().str.strip()
        if 'income_level' in df.columns:
            df['income_level'] = df['income_level'].astype(str).str.lower().str.strip()

        # Normalize boolean target if present
        if 'is_licensed' in df.columns:
            if df['is_licensed'].dtype != bool:
                df['is_licensed'] = df['is_licensed'].map({
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                    '1': True,
                    '0': False,
                    'yes': True,
                    'no': False,
                    'true': True,
                    'false': False
                }).fillna(False).astype(bool)

        return df

    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the vehicle licensing data"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            df = self._standardize_columns(df)
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
        date_columns = ['registration_date', 'last_license_renewal', 'license_expiry_date', 'data_collection_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 4. Remove records with impossible dates (future registration dates, etc.)
        if 'data_collection_date' in df.columns:
            df = df[df['data_collection_date'] <= pd.Timestamp.now()]

        # 5. Handle agent sync delays - if payment is not through agent, sync delay should be 0
        if 'agent_sync_delay' in df.columns and ('payment_mode' in df.columns or 'agent_service_used' in df.columns):
            agent_payment_mask = pd.Series(False, index=df.index)
            if 'payment_mode' in df.columns:
                agent_payment_mask = agent_payment_mask | (df['payment_mode'] == 'agent')
            if 'agent_service_used' in df.columns:
                agent_payment_mask = agent_payment_mask | df['agent_service_used'].astype(str).str.lower().isin(['agent', 'in-person', 'in person'])
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
        """Advanced feature engineering for model training"""
        logger.info("Engineering advanced features...")

        # Basic temporal features
        if 'days_since_renewal' not in df.columns and 'last_license_renewal' in df.columns:
            df['days_since_renewal'] = (pd.Timestamp.now() - df['last_license_renewal']).dt.days
            df['days_since_renewal'] = df['days_since_renewal'].fillna(365)
        
        # Use Days Since Last Renewal from enriched data if available
        if 'Days Since Last Renewal' in df.columns:
            df['days_since_renewal'] = pd.to_numeric(df['Days Since Last Renewal'], errors='coerce').fillna(df.get('days_since_renewal', 365))

        # Vehicle age
        if 'vehicle_age_years' not in df.columns:
            if 'Year' in df.columns:
                current_year = pd.Timestamp.now().year
                df['vehicle_age_years'] = current_year - pd.to_numeric(df['Year'], errors='coerce')
                df['vehicle_age_years'] = df['vehicle_age_years'].clip(lower=0).fillna(df['vehicle_age_years'].median())

        # Renewal frequency
        if 'renewal_frequency' not in df.columns:
            if 'total_renewals' in df.columns and 'vehicle_age_years' in df.columns:
                df['renewal_frequency'] = df['total_renewals'] / (df['vehicle_age_years'] + 1)
            elif 'Average Renewal Lag Days' in df.columns:
                # Inverse relationship: lower lag = higher frequency
                max_lag = df['Average Renewal Lag Days'].max() if 'Average Renewal Lag Days' in df.columns else 365
                df['renewal_frequency'] = 1.0 - (pd.to_numeric(df.get('Average Renewal Lag Days', 0), errors='coerce') / (max_lag + 1))

        # Late renewal rate
        if 'late_renewal_rate' not in df.columns:
            if 'Number of Late Renewals in Last 3 Years' in df.columns:
                late_renewals = pd.to_numeric(df['Number of Late Renewals in Last 3 Years'], errors='coerce').fillna(0)
                df['late_renewal_rate'] = late_renewals / 3.0
            elif 'late_renewals_count' in df.columns and 'total_renewals' in df.columns:
                df['late_renewal_rate'] = df['late_renewals_count'] / (df['total_renewals'] + 1)

        # Temporal features
        if 'renewal_month' not in df.columns and 'last_license_renewal' in df.columns:
            df['renewal_month'] = df['last_license_renewal'].dt.month
            df['renewal_quarter'] = df['last_license_renewal'].dt.quarter
        elif 'Month' in df.columns:
            df['renewal_month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['renewal_quarter'] = pd.to_numeric(df.get('Quarter', df['renewal_month'] // 3 + 1), errors='coerce')

        # Payment digitalization score
        if 'payment_digital_score' not in df.columns:
            if 'payment_mode' in df.columns:
                digital_mapping = {'online': 1.0, 'mobile_money': 0.8, 'agent': 0.4, 'cash': 0.0}
                df['payment_digital_score'] = df['payment_mode'].map(digital_mapping)
            elif 'Preferred Payment Mode' in df.columns:
                digital_mapping = {
                    'Online': 1.0, 'Mobile Money': 0.8, 'Bank Transfer': 0.6,
                    'Agent': 0.4, 'In-person': 0.3, 'Cash': 0.0
                }
                df['payment_digital_score'] = df['Preferred Payment Mode'].map(digital_mapping).fillna(0.5)

        # Regional risk score
        if 'region_risk_score' not in df.columns:
            if 'region' in df.columns or 'Region' in df.columns:
                region_col = 'region' if 'region' in df.columns else 'Region'
                region_risk = {'urban': 0.3, 'Urban': 0.3, 'peri_urban': 0.5, 'Peri-urban': 0.5, 'rural': 0.7, 'Rural': 0.7}
                df['region_risk_score'] = df[region_col].map(region_risk).fillna(0.5)

        # Agent efficiency
        if 'agent_efficiency' not in df.columns:
            if 'agent_sync_delay' in df.columns:
                max_delay = df['agent_sync_delay'].max()
                if pd.isna(max_delay) or max_delay <= 0:
                    max_delay = 1
                df['agent_efficiency'] = 1 - (df['agent_sync_delay'] / (max_delay + 1))
            elif 'Agent Synchronization Lag' in df.columns:
                sync_lag = pd.to_numeric(df['Agent Synchronization Lag'], errors='coerce').fillna(0)
                max_delay = sync_lag.max() if (pd.notna(sync_lag.max()) and sync_lag.max() > 0) else 1
                df['agent_efficiency'] = 1 - (sync_lag / (max_delay + 1))

        # ADVANCED FEATURES - Interaction and polynomial features
        # Days since renewal squared (non-linear relationship)
        if 'days_since_renewal' in df.columns:
            df['days_since_renewal_sq'] = df['days_since_renewal'] ** 2
            df['days_since_renewal_log'] = np.log1p(df['days_since_renewal'])

        # Interaction: Payment digital score * Region risk
        if 'payment_digital_score' in df.columns and 'region_risk_score' in df.columns:
            df['payment_region_interaction'] = df['payment_digital_score'] * df['region_risk_score']

        # Interaction: Late renewal rate * Days since renewal
        if 'late_renewal_rate' in df.columns and 'days_since_renewal' in df.columns:
            df['late_renewal_days_interaction'] = df['late_renewal_rate'] * df['days_since_renewal']

        # Compliance score (combination of multiple factors)
        compliance_factors = []
        if 'late_renewal_rate' in df.columns:
            compliance_factors.append(1 - df['late_renewal_rate'])
        if 'payment_digital_score' in df.columns:
            compliance_factors.append(df['payment_digital_score'])
        if 'agent_efficiency' in df.columns:
            compliance_factors.append(df['agent_efficiency'])
        if compliance_factors:
            df['compliance_score'] = pd.concat(compliance_factors, axis=1).mean(axis=1)

        # Risk accumulation score
        risk_factors = []
        if 'days_since_renewal' in df.columns:
            risk_factors.append(df['days_since_renewal'] / 365.0)  # Normalize to years
        if 'late_renewal_rate' in df.columns:
            risk_factors.append(df['late_renewal_rate'])
        if 'region_risk_score' in df.columns:
            risk_factors.append(df['region_risk_score'])
        if 'previous_violations' in df.columns:
            violations = pd.to_numeric(df['previous_violations'], errors='coerce').fillna(0)
            max_violations = violations.max()
            if pd.isna(max_violations) or max_violations <= 0:
                max_violations = 1
            risk_factors.append(violations / (max_violations + 1))
        if risk_factors:
            df['risk_accumulation_score'] = pd.concat(risk_factors, axis=1).mean(axis=1)

        # Time-based cyclical encoding for month
        if 'renewal_month' in df.columns:
            df['renewal_month_sin'] = np.sin(2 * np.pi * df['renewal_month'] / 12)
            df['renewal_month_cos'] = np.cos(2 * np.pi * df['renewal_month'] / 12)

        # Fine amount normalized
        if 'fine_amount' in df.columns:
            fine = pd.to_numeric(df['fine_amount'], errors='coerce').fillna(0)
            max_fine = fine.max() if fine.max() > 0 else 1
            df['fine_amount_normalized'] = fine / max_fine
        elif 'Fine Amount' in df.columns:
            fine = pd.to_numeric(df['Fine Amount'], errors='coerce').fillna(0)
            max_fine = fine.max() if fine.max() > 0 else 1
            df['fine_amount_normalized'] = fine / max_fine
        else:
            # Default if fine amount not available
            df['fine_amount_normalized'] = 0.0

        # Ensure critical features exist with defaults if they couldn't be computed
        # Create base features first
        if 'days_since_renewal' not in df.columns:
            df['days_since_renewal'] = 365.0  # Default: 1 year
        if 'late_renewal_rate' not in df.columns:
            df['late_renewal_rate'] = 0.0  # Default: no late renewals
        if 'agent_efficiency' not in df.columns:
            df['agent_efficiency'] = 0.5  # Default: medium efficiency
        if 'payment_digital_score' not in df.columns:
            df['payment_digital_score'] = 0.5  # Default: medium digitalization
        if 'region_risk_score' not in df.columns:
            df['region_risk_score'] = 0.5  # Default: medium risk
        
        # Create polynomial features from days_since_renewal (ensure it exists first)
        if 'days_since_renewal' in df.columns:
            if 'days_since_renewal_sq' not in df.columns:
                df['days_since_renewal_sq'] = df['days_since_renewal'] ** 2
            if 'days_since_renewal_log' not in df.columns:
                df['days_since_renewal_log'] = np.log1p(df['days_since_renewal'])
        
        # Create interaction features (ensure base features exist)
        if 'payment_region_interaction' not in df.columns:
            payment_score = df.get('payment_digital_score', pd.Series([0.5] * len(df)))
            region_score = df.get('region_risk_score', pd.Series([0.5] * len(df)))
            df['payment_region_interaction'] = payment_score * region_score
        
        if 'late_renewal_days_interaction' not in df.columns:
            late_rate = df.get('late_renewal_rate', pd.Series([0.0] * len(df)))
            days_renewal = df.get('days_since_renewal', pd.Series([365.0] * len(df)))
            df['late_renewal_days_interaction'] = late_rate * days_renewal
        
        # Create composite scores
        if 'compliance_score' not in df.columns:
            compliance_factors = []
            if 'late_renewal_rate' in df.columns:
                compliance_factors.append(1 - df['late_renewal_rate'])
            if 'payment_digital_score' in df.columns:
                compliance_factors.append(df['payment_digital_score'])
            if 'agent_efficiency' in df.columns:
                compliance_factors.append(df['agent_efficiency'])
            if compliance_factors:
                df['compliance_score'] = pd.concat(compliance_factors, axis=1).mean(axis=1)
            else:
                df['compliance_score'] = 0.5  # Default
        
        if 'risk_accumulation_score' not in df.columns:
            risk_factors = []
            if 'days_since_renewal' in df.columns:
                risk_factors.append(df['days_since_renewal'] / 365.0)
            if 'late_renewal_rate' in df.columns:
                risk_factors.append(df['late_renewal_rate'])
            if 'region_risk_score' in df.columns:
                risk_factors.append(df['region_risk_score'])
            if 'previous_violations' in df.columns:
                violations = pd.to_numeric(df['previous_violations'], errors='coerce').fillna(0)
                max_violations = violations.max()
                if pd.isna(max_violations) or max_violations <= 0:
                    max_violations = 1
                risk_factors.append(violations / max_violations)
            if risk_factors:
                df['risk_accumulation_score'] = pd.concat(risk_factors, axis=1).mean(axis=1)
            else:
                df['risk_accumulation_score'] = 0.5  # Default
        
        # Create cyclical encoding for month
        if 'renewal_month_sin' not in df.columns:
            if 'renewal_month' in df.columns:
                df['renewal_month_sin'] = np.sin(2 * np.pi * df['renewal_month'] / 12)
            else:
                df['renewal_month_sin'] = 0.0  # Default
        if 'renewal_month_cos' not in df.columns:
            if 'renewal_month' in df.columns:
                df['renewal_month_cos'] = np.cos(2 * np.pi * df['renewal_month'] / 12)
            else:
                df['renewal_month_cos'] = 1.0  # Default

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def prepare_features_target(self, df, target_column='is_licensed'):
        """Prepare features and target for model training"""

        # Define feature columns - including all advanced features (matching notebook)
        feature_cols = [
            # Basic features
            'days_since_renewal', 'Days Since Last Renewal', 'vehicle_age_years',
            'Number of Late Renewals in Last 3 Years', 'Average Renewal Lag Days',
            'renewal_frequency', 'late_renewal_rate', 'payment_digital_score',
            'region_risk_score', 'agent_efficiency', 'renewal_month', 'renewal_quarter',
            'Month', 'Quarter',
            # Advanced features
            'days_since_renewal_sq', 'days_since_renewal_log',
            'payment_region_interaction', 'late_renewal_days_interaction',
            'compliance_score', 'risk_accumulation_score',
            'renewal_month_sin', 'renewal_month_cos',
            'fine_amount_normalized', 'Fine Amount',
            'Previous Violations', 'Predictive Score', 'total_vehicles_owned',
            'Number of Vehicles Owned', 'Agent Hours Used', 'User Feedback Score'
        ]

        # Add categorical features with encoding
        categorical_features = []

        # Vehicle type encoding
        if 'vehicle_type' in df.columns:
            # Store unique categories for consistent encoding during prediction
            if 'vehicle_type' not in self.categorical_categories:
                self.categorical_categories['vehicle_type'] = sorted(df['vehicle_type'].dropna().unique().tolist())
            vehicle_type_encoded = pd.get_dummies(df['vehicle_type'], prefix='vehicle_type')
            df = pd.concat([df, vehicle_type_encoded], axis=1)
            # Store the exact column names created by pd.get_dummies for this categorical
            if 'vehicle_type' not in self.categorical_categories:
                self.categorical_categories['vehicle_type'] = []
            # Store both the category values AND the exact column names
            if not hasattr(self, '_categorical_column_names'):
                self._categorical_column_names = {}
            self._categorical_column_names['vehicle_type'] = vehicle_type_encoded.columns.tolist()
            categorical_features.extend(vehicle_type_encoded.columns.tolist())

        # Region encoding
        if 'region' in df.columns:
            if 'region' not in self.categorical_categories:
                self.categorical_categories['region'] = sorted(df['region'].dropna().unique().tolist())
            region_encoded = pd.get_dummies(df['region'], prefix='region')
            df = pd.concat([df, region_encoded], axis=1)
            if not hasattr(self, '_categorical_column_names'):
                self._categorical_column_names = {}
            self._categorical_column_names['region'] = region_encoded.columns.tolist()
            categorical_features.extend(region_encoded.columns.tolist())

        # Payment mode encoding
        if 'payment_mode' in df.columns:
            if 'payment_mode' not in self.categorical_categories:
                self.categorical_categories['payment_mode'] = sorted(df['payment_mode'].dropna().unique().tolist())
            payment_encoded = pd.get_dummies(df['payment_mode'], prefix='payment')
            df = pd.concat([df, payment_encoded], axis=1)
            if not hasattr(self, '_categorical_column_names'):
                self._categorical_column_names = {}
            self._categorical_column_names['payment_mode'] = payment_encoded.columns.tolist()
            categorical_features.extend(payment_encoded.columns.tolist())

        # Income level encoding
        if 'income_level' in df.columns:
            if 'income_level' not in self.categorical_categories:
                self.categorical_categories['income_level'] = sorted(df['income_level'].dropna().unique().tolist())
            income_encoded = pd.get_dummies(df['income_level'], prefix='income')
            df = pd.concat([df, income_encoded], axis=1)
            if not hasattr(self, '_categorical_column_names'):
                self._categorical_column_names = {}
            self._categorical_column_names['income_level'] = income_encoded.columns.tolist()
            categorical_features.extend(income_encoded.columns.tolist())

        # Combine all features
        all_features = feature_cols + categorical_features

        # Filter existing columns
        available_features = [col for col in all_features if col in df.columns]
        self.feature_columns = available_features
        # store full feature set (before any selection)
        self.all_feature_columns = available_features.copy()

        # Prepare X and y
        X = df[available_features]
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        target_series = df[target_column].astype(bool)
        y = (~target_series).astype(int)  # Predict 1 for unlicensed

        logger.info(f"Prepared {len(available_features)} features for training")
        logger.info(f"Target distribution - Unlicensed: {y.sum()}, Licensed: {(~y).sum()}")

        return X, y

    def train_model(self, X, y, model_type='xgboost', test_size=0.2, random_state=42):
        """Advanced model training with feature selection, SMOTE, cross-validation, and hyperparameter tuning"""
        logger.info(f"Training {model_type} model with advanced techniques...")
        logger.info(f"Initial dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: Unlicensed={y.sum()}, Licensed={(~y.astype(bool)).sum()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        original_feature_columns = self.feature_columns.copy() if self.feature_columns else None

        # Feature Selection - Select top K features using mutual information
        if self.use_feature_selection and X_train.shape[1] > 30:
            logger.info("Performing feature selection...")
            k_best = min(50, X_train.shape[1] // 2)  # Select top 50% or 50 features, whichever is smaller
            selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            self.feature_selector = selector
            # Update feature columns to selected features
            # Validate that feature_columns has enough elements before indexing
            if not self.feature_columns or len(self.feature_columns) != X_train.shape[1]:
                raise ValueError(f"Feature columns mismatch: expected {X_train.shape[1]} columns, but feature_columns has {len(self.feature_columns) if self.feature_columns else 0} elements")
            selected_indices = selector.get_support(indices=True)
            # Ensure all indices are within bounds
            max_index = len(self.feature_columns) - 1
            valid_indices = [i for i in selected_indices if 0 <= i <= max_index]
            if len(valid_indices) != len(selected_indices):
                logger.warning(f"Some feature indices were out of range. Using {len(valid_indices)} valid indices out of {len(selected_indices)}")
            selected_features = [self.feature_columns[i] for i in valid_indices]
            self.feature_columns = selected_features
            if original_feature_columns:
                self.all_feature_columns = original_feature_columns
            logger.info(f"Selected {len(selected_features)} features from {X_train.shape[1]} original features")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            self.feature_selector = None
            if original_feature_columns and not self.all_feature_columns:
                self.all_feature_columns = original_feature_columns

        # Apply SMOTE for class imbalance (only on training set)
        if self.use_smote and IMBLEARN_AVAILABLE:
            try:
                logger.info("Applying SMOTE for class imbalance...")
                pos_count = (y_train == 1).sum()
                if pos_count > 1:  # Need at least 2 samples for SMOTE
                    k_neighbors = min(5, pos_count - 1)
                    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
                    logger.info(f"After SMOTE: {len(X_train_balanced)} samples (was {len(X_train_selected)})")
                    logger.info(f"Balanced class distribution: Unlicensed={(y_train_balanced == 1).sum()}, Licensed={(y_train_balanced == 0).sum()}")
                else:
                    logger.warning("Not enough positive samples for SMOTE. Using original data.")
                    X_train_balanced, y_train_balanced = X_train_selected, y_train
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Using original data.")
                X_train_balanced, y_train_balanced = X_train_selected, y_train
        else:
            if self.use_smote and not IMBLEARN_AVAILABLE:
                logger.warning("SMOTE requested but imbalanced-learn not installed. Using original data.")
            X_train_balanced, y_train_balanced = X_train_selected, y_train

        # Scale features (for models that need it)
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test_selected)

        # Initialize and train model based on type
        self.model_type = model_type

        if model_type == 'logistic_regression':
            # Hyperparameter tuning for Logistic Regression
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train_balanced)
            self.model = grid_search.best_estimator_
            logger.info(f"Best LR params: {grid_search.best_params_}")

        elif model_type == 'decision_tree':
            # Hyperparameter tuning for Decision Tree
            param_grid = {
                'max_depth': [8, 12, 15, 20],
                'min_samples_leaf': [5, 10, 15, 20],
                'min_samples_split': [10, 20, 30]
            }
            base_model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_balanced, y_train_balanced)
            self.model = grid_search.best_estimator_
            logger.info(f"Best DT params: {grid_search.best_params_}")

        elif model_type == 'random_forest':
            # Enhanced Random Forest with hyperparameter tuning
            pos_count = (y_train_balanced == 1).sum()
            neg_count = (y_train_balanced == 0).sum()
            class_weight = 'balanced' if abs(pos_count - neg_count) / len(y_train_balanced) > 0.1 else None
            
            param_grid = {
                'n_estimators': [300, 500, 700],
                'max_depth': [10, 15, 20],
                'min_samples_leaf': [2, 3, 5],
                'min_samples_split': [5, 10, 15]
            }
            base_model = RandomForestClassifier(
                random_state=random_state,
                class_weight=class_weight,
                n_jobs=-1
            )
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_balanced, y_train_balanced)
            self.model = grid_search.best_estimator_
            logger.info(f"Best RF params: {grid_search.best_params_}")

        elif model_type == 'xgboost':
            # Enhanced XGBoost with comprehensive hyperparameter tuning
            pos_count = (y_train_balanced == 1).sum()
            neg_count = (y_train_balanced == 0).sum()
            scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

            param_grid = {
                'n_estimators': [500, 800, 1000],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.03, 0.05],
                'subsample': [0.8, 0.85, 0.9],
                'colsample_bytree': [0.8, 0.85, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_lambda': [1.0, 1.5, 2.0],
                'min_child_weight': [1, 3, 5]
            }
            base_model = xgb.XGBClassifier(
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                n_jobs=-1
            )
            # Use RandomizedSearchCV for faster training with large param grid
            from sklearn.model_selection import RandomizedSearchCV
            grid_search = RandomizedSearchCV(
                base_model, param_grid, cv=3, scoring='f1', 
                n_jobs=-1, verbose=1, n_iter=30, random_state=random_state
            )
            grid_search.fit(X_train_balanced, y_train_balanced)
            self.model = grid_search.best_estimator_
            logger.info(f"Best XGB params: {grid_search.best_params_}")

        elif model_type == 'ensemble':
            # Ensemble of multiple models
            logger.info("Training ensemble model...")
            pos_count = (y_train_balanced == 1).sum()
            neg_count = (y_train_balanced == 0).sum()
            scale_pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

            # Individual models
            rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=3, 
                                             random_state=random_state, n_jobs=-1)
            xgb_model = xgb.XGBClassifier(n_estimators=800, max_depth=7, learning_rate=0.03,
                                         scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1)
            gb_model = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                                                  random_state=random_state)

            # Voting ensemble
            self.model = VotingClassifier(
                estimators=[('rf', rf_model), ('xgb', xgb_model), ('gb', gb_model)],
                voting='soft',
                weights=[1, 2, 1]  # XGBoost gets more weight
            )
            self.model.fit(X_train_balanced, y_train_balanced)

        # Cross-validation evaluation
        logger.info("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        if model_type in ['logistic_regression']:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train_balanced, 
                                       cv=cv, scoring='f1', n_jobs=-1)
        else:
            cv_scores = cross_val_score(self.model, X_train_balanced, y_train_balanced, 
                                       cv=cv, scoring='f1', n_jobs=-1)
        logger.info(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Evaluate model with threshold optimization on test set
        if model_type in ['logistic_regression']:
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]

        # Choose threshold maximizing F1
        best_f1 = -1.0
        best_thr = 0.5
        for thr in np.linspace(0.2, 0.8, 50):  # More granular threshold search
            y_pred_thr = (y_pred_proba >= thr).astype(int)
            f1 = f1_score(y_test, y_pred_thr)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        self.threshold_ = best_thr

        # Calculate predictions with optimal threshold
        y_pred = (y_pred_proba >= self.threshold_).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            'matrix': cm.tolist(),
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'threshold': float(self.threshold_),
            'confusion_matrix': cm_dict,
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std())
        }

        logger.info(f"Model training complete. Test Metrics: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        return metrics

    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            importance_scores = np.abs(np.array(coef, dtype=float))
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
        meta_file = os.path.join(model_path, f"{model_name}_meta.joblib")
        selector_file = os.path.join(model_path, f"{model_name}_selector.joblib")
        categories_file = os.path.join(model_path, f"{model_name}_categories.joblib")

        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_columns, features_file)
        meta_payload = {
            'threshold': self.threshold_,
            'model_type': self.model_type,
            'all_features': self.all_feature_columns
        }
        joblib.dump(meta_payload, meta_file)
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, selector_file)
        if self.categorical_categories:
            joblib.dump(self.categorical_categories, categories_file)
        # Save exact categorical column names created by pd.get_dummies
        if hasattr(self, '_categorical_column_names'):
            categorical_cols_file = os.path.join(model_path, f"{model_name}_categorical_cols.joblib")
            joblib.dump(self._categorical_column_names, categorical_cols_file)

        logger.info(f"Model saved to {model_file}")
        return model_file

    def load_model(self, model_path, model_name):
        """Load a saved model"""
        model_file = os.path.join(model_path, f"{model_name}.joblib")
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.joblib")
        features_file = os.path.join(model_path, f"{model_name}_features.joblib")
        meta_file = os.path.join(model_path, f"{model_name}_meta.joblib")
        selector_file = os.path.join(model_path, f"{model_name}_selector.joblib")
        categories_file = os.path.join(model_path, f"{model_name}_categories.joblib")
        categorical_cols_file = os.path.join(model_path, f"{model_name}_categorical_cols.joblib")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_columns = joblib.load(features_file)
        if os.path.exists(meta_file):
            meta = joblib.load(meta_file)
            self.threshold_ = float(meta.get('threshold', 0.5))
            self.model_type = meta.get('model_type')
            self.all_feature_columns = meta.get('all_features')
        if os.path.exists(selector_file):
            self.feature_selector = joblib.load(selector_file)
        else:
            self.feature_selector = None
        # Backward compatibility: if all_feature_columns not stored, default to feature_columns
        if not self.all_feature_columns and self.feature_columns:
            self.all_feature_columns = self.feature_columns.copy()
        if os.path.exists(categories_file):
            self.categorical_categories = joblib.load(categories_file)
        else:
            self.categorical_categories = {}
        # Load exact categorical column names if available
        if os.path.exists(categorical_cols_file):
            self._categorical_column_names = joblib.load(categorical_cols_file)
        else:
            self._categorical_column_names = {}

        logger.info(f"Model loaded from {model_file}")

    def _ensure_all_features_exist(self, df):
        """Ensure all expected features exist in the DataFrame, creating missing ones with defaults"""
        # Use all_feature_columns (before selection) if available, otherwise feature_columns
        # This ensures we create ALL features that might be needed, not just selected ones
        feature_list = self.all_feature_columns or self.feature_columns
        if not feature_list:
            return df
        
        df = df.copy()
        
        # First, ensure all base features that might be needed are created
        # This includes polynomial and interaction features
        if 'days_since_renewal' not in df.columns:
            df['days_since_renewal'] = 365.0
        if 'days_since_renewal' in df.columns:
            if 'days_since_renewal_sq' not in df.columns:
                df['days_since_renewal_sq'] = df['days_since_renewal'] ** 2
            if 'days_since_renewal_log' not in df.columns:
                df['days_since_renewal_log'] = np.log1p(df['days_since_renewal'])
        
        if 'late_renewal_rate' not in df.columns:
            df['late_renewal_rate'] = 0.0
        if 'payment_digital_score' not in df.columns:
            df['payment_digital_score'] = 0.5
        if 'region_risk_score' not in df.columns:
            df['region_risk_score'] = 0.5
        
        if 'late_renewal_days_interaction' not in df.columns:
            late_rate = df.get('late_renewal_rate', pd.Series([0.0] * len(df)))
            days_renewal = df.get('days_since_renewal', pd.Series([365.0] * len(df)))
            df['late_renewal_days_interaction'] = late_rate * days_renewal
        
        # CRITICAL: Now ensure ALL expected features exist (from the full feature list)
        # This includes all dummy variables and engineered features
        missing_features = []
        for feature in feature_list:
            if feature not in df.columns:
                missing_features.append(feature)
                df[feature] = 0
        
        if missing_features:
            logger.info(f"Created {len(missing_features)} missing features with default values: {missing_features[:10]}...")
        
        return df

    def _encode_categoricals_consistently(self, df):
        """Encode categorical features consistently using stored categories"""
        df = df.copy()
        
        # Determine prefix mapping (matching what pd.get_dummies uses)
        prefix_map = {
            'vehicle_type': 'vehicle_type',
            'region': 'region',
            'payment_mode': 'payment',
            'income_level': 'income'
        }
        
        # If we have exact column names from training, use those
        if hasattr(self, '_categorical_column_names') and self._categorical_column_names:
            # Use exact column names that were created during training
            for cat_col, exact_column_names in self._categorical_column_names.items():
                prefix = prefix_map.get(cat_col, cat_col)
                
                # Normalize the column values for comparison if it exists
                if cat_col in df.columns:
                    df[cat_col] = df[cat_col].astype(str).str.lower().str.strip()
                else:
                    df[cat_col] = ''
                
                # Create each exact column name that was created during training
                for exact_col_name in exact_column_names:
                    # Extract the category value from the column name (remove prefix_)
                    category_value = exact_col_name.replace(f"{prefix}_", "", 1)
                    normalized_cat = category_value.lower().strip()
                    
                    # Create the exact column name
                    if cat_col in df.columns and len(df) > 0:
                        df[exact_col_name] = (df[cat_col] == normalized_cat).astype(int)
                    else:
                        df[exact_col_name] = 0
        else:
            # Fallback: create dummy variables using stored categories
            for cat_col, expected_categories in self.categorical_categories.items():
                prefix = prefix_map.get(cat_col, cat_col)
                
                # Normalize the column values for comparison if it exists
                if cat_col in df.columns:
                    df[cat_col] = df[cat_col].astype(str).str.lower().str.strip()
                else:
                    df[cat_col] = ''
                
                # Create dummy variables for all expected categories
                # Try to match what pd.get_dummies creates
                for category in expected_categories:
                    category_str = str(category)
                    normalized_cat = category_str.lower().strip()
                    
                    # pd.get_dummies typically creates: prefix_category (with spaces/underscores as-is)
                    # But it might normalize - try multiple variations
                    dummy_cols = [
                        f"{prefix}_{category_str}",  # Exact
                        f"{prefix}_{normalized_cat}",  # Lowercased
                        f"{prefix}_{normalized_cat.replace(' ', '_')}",  # Spaces to underscores
                    ]
                    
                    for dummy_col in dummy_cols:
                        if cat_col in df.columns and len(df) > 0:
                            df[dummy_col] = (df[cat_col] == normalized_cat).astype(int)
                        else:
                            df[dummy_col] = 0
        
        return df

    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Always work with a DataFrame
        if not isinstance(X, pd.DataFrame):
            base_features = self.all_feature_columns or self.feature_columns
            if not base_features:
                raise ValueError("Feature list not available to construct DataFrame for prediction")
            X = pd.DataFrame(X, columns=base_features)
        else:
            X = X.copy()

        # Encode categorical variables consistently FIRST
        if self.categorical_categories:
            X = self._encode_categoricals_consistently(X)

        # Ensure engineered features exist (uses full list when available)
        X = self._ensure_all_features_exist(X)

        # Handle feature selection if it was used during training
        if self.feature_selector is not None:
            # Feature selection was used - we need ALL original features first
            all_original_features = self.all_feature_columns or self.feature_columns
            if not all_original_features:
                raise ValueError("Feature selector present but original feature list not available")
            
            logger.info(f"Feature selection used: need {len(all_original_features)} original features for selector")
            logger.info(f"Current X has {X.shape[1]} features, columns: {list(X.columns)[:10]}...")
            
            # CRITICAL: Ensure X has EXACTLY all original features in the EXACT order
            # This is what the feature selector was trained on
            missing_original = [f for f in all_original_features if f not in X.columns]
            if missing_original:
                logger.warning(f"Missing {len(missing_original)} original features: {missing_original[:5]}... Creating with defaults.")
                for feature in missing_original:
                    X[feature] = 0
            
            # Reindex to ensure correct order and all features present
            X = X.reindex(columns=all_original_features, fill_value=0)
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Validate we have the right number of features
            if X.shape[1] != len(all_original_features):
                raise ValueError(f"Feature count mismatch: X has {X.shape[1]} features but need {len(all_original_features)}. "
                               f"Missing: {set(all_original_features) - set(X.columns)}")
            
            # Validate column order matches exactly
            if list(X.columns) != all_original_features:
                logger.warning(f"Column order mismatch. Reordering to match training order.")
                X = X.reindex(columns=all_original_features, fill_value=0)
            
            logger.info(f"X prepared with {X.shape[1]} features for feature selector")
            
            # Get selected feature names (what the model actually expects after selection)
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                selected_features = list(self.model.feature_names_in_)
            else:
                # Fallback: get selected features from selector
                selected_indices = self.feature_selector.get_support(indices=True)
                # Validate indices are within bounds
                max_index = len(all_original_features) - 1
                valid_indices = [i for i in selected_indices if 0 <= i <= max_index]
                if len(valid_indices) != len(selected_indices):
                    logger.warning(f"Some feature indices were out of range. Using {len(valid_indices)} valid indices out of {len(selected_indices)}")
                selected_features = [all_original_features[i] for i in valid_indices]
            
            logger.info(f"Model expects {len(selected_features)} selected features after selection")
            
            # STEP 2: Apply feature selection (reduces from all_original_features to selected_features)
            # Convert to numpy array for selector (it expects the exact shape it was trained on)
            X_array = X.values
            logger.info(f"Applying feature selector: input shape {X_array.shape}, expecting {len(all_original_features)} features")
            
            try:
                X_selected = self.feature_selector.transform(X_array)
            except ValueError as e:
                logger.error(f"Feature selector error: {e}")
                logger.error(f"X shape: {X_array.shape}, expected: ({X_array.shape[0]}, {len(all_original_features)})")
                logger.error(f"X columns: {list(X.columns)[:20]}")
                raise ValueError(f"Feature selector failed: {e}. X has {X_array.shape[1]} features but selector expects {len(all_original_features)}. "
                               f"Ensure all features from training are present.")
            
            # STEP 3: Convert back to DataFrame with selected feature names
            X = pd.DataFrame(X_selected, columns=selected_features)
            
            # Final check
            if list(X.columns) != selected_features:
                logger.error(f"Feature mismatch after selection! Reordering...")
                X = X.reindex(columns=selected_features, fill_value=0)
        else:
            # No feature selection - model expects all features
            if hasattr(self.model, 'feature_names_in_') and self.model.feature_names_in_ is not None:
                expected_features = list(self.model.feature_names_in_)
            elif self.feature_columns:
                expected_features = self.feature_columns
            else:
                raise ValueError("Cannot determine expected features for model")
            
            logger.info(f"No feature selection: model expects {len(expected_features)} features")
            
            # Ensure all expected features exist
            missing_features = [f for f in expected_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features. Creating with defaults.")
                for feature in missing_features:
                    X[feature] = 0
            
            # Reindex to ensure correct order
            if isinstance(X, pd.DataFrame):
                X = X.reindex(columns=expected_features, fill_value=0)
                X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Final validation
                if list(X.columns) != expected_features:
                    logger.error(f"Feature order mismatch! Reordering...")
                    X = X.reindex(columns=expected_features, fill_value=0)

        # Scale if needed
        if self.model_type == 'logistic_regression':
            # Logistic regression needs numpy array for scaling
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            X_scaled = self.scaler.transform(X_array)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            # For XGBoost and other tree models
            # If X is DataFrame, XGBoost will validate feature names match feature_names_in_
            # If X is numpy array, it uses position-based matching
            # Since we've ensured the DataFrame has the right columns in the right order,
            # we can pass it as DataFrame and XGBoost will validate
            if isinstance(X, pd.DataFrame):
                probabilities = self.model.predict_proba(X)[:, 1]
            else:
                # Numpy array - order must match training order exactly
                probabilities = self.model.predict_proba(X)[:, 1]

        # Apply threshold
        predictions = (probabilities >= self.threshold_).astype(int)

        return predictions, probabilities