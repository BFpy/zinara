import os
import pandas as pd
import numpy as np
from datetime import datetime

RAW_PATH = os.path.join(os.path.dirname(__file__), "raw", "zinara_vehicle_licensing_data_2015_2024.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "zinara_vehicle_licensing_enriched.csv")

CANONICAL_PAYMENT = {
	'online': 'Online',
	'mobile money': 'Mobile Money',
	'mobile_money': 'Mobile Money',
	'agent': 'Agent',
	'in-person': 'In-person',
	'in person': 'In-person',
	'cash': 'Cash',
	'bank transfer': 'Bank Transfer',
	'bank_transfer': 'Bank Transfer'
}

REGION_SET = {'Urban', 'Peri-urban', 'Rural'}

def parse_date(series: pd.Series) -> pd.Series:
	# Handle both '-' and '/' and coerce invalids
	ser = pd.to_datetime(series.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip(), errors='coerce')
	return ser

def safe_int(x, default=np.nan):
	try:
		return int(x)
	except Exception:
		return default

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure required columns exist, creating with NaNs if missing."""
	required = [
		'Vehicle ID', 'Owner ID', 'Make', 'Model', 'Year', 'Vehicle Type',
		'License Status', 'Last Renewal Date', 'Expiration Date', 'Fine Amount',
		'Agent Service Used', 'Online Platform Used', 'Compliance History',
		'Geographic Location', 'Region', 'Income Level', 'Number of Vehicles Owned',
		'Previous Violations', 'Agent Hours Used', 'User Feedback Score',
		'Predictive Score', 'Preferred Payment Mode', 'Date of Data Collection',
		'Days Since Last Renewal', 'Number of Late Renewals in Last 3 Years',
		'Average Renewal Lag Days', 'Agent Synchronization Lag', 'Month', 'Quarter'
	]
	for col in required:
		if col not in df.columns:
			df[col] = np.nan
	return df

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
	# Normalize Preferred Payment Mode
	if 'Preferred Payment Mode' in df.columns:
		df['Preferred Payment Mode'] = (
			df['Preferred Payment Mode']
			.astype(str)
			.str.strip()
			.str.replace('_', ' ')
			.str.lower()
			.map(CANONICAL_PAYMENT)
			.fillna(df['Preferred Payment Mode'])
		)

	# Normalize Region to expected set
	if 'Region' in df.columns:
		df['Region'] = df['Region'].astype(str).str.strip().str.title()
		df.loc[~df['Region'].isin(REGION_SET), 'Region'] = np.nan

	# Vehicle Type canonical spacing/case
	if 'Vehicle Type' in df.columns:
		df['Vehicle Type'] = df['Vehicle Type'].astype(str).str.strip().str.title()

	# Income level title case
	if 'Income Level' in df.columns:
		df['Income Level'] = df['Income Level'].astype(str).str.strip().str.title()

	return df

def derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
	# Parse dates
	df['Last Renewal Date'] = parse_date(df['Last Renewal Date'])
	df['Expiration Date'] = parse_date(df['Expiration Date'])
	df['Date of Data Collection'] = parse_date(df['Date of Data Collection'])

	# Fill data collection date with today if missing
	df['Date of Data Collection'] = df['Date of Data Collection'].fillna(pd.Timestamp.today().normalize())

	# Recompute Days Since Last Renewal
	mask_last = df['Last Renewal Date'].notna()
	df.loc[mask_last, 'Days Since Last Renewal'] = (df.loc[mask_last, 'Date of Data Collection'] - df.loc[mask_last, 'Last Renewal Date']).dt.days.clip(lower=0)

	# Month and Quarter from Date of Data Collection
	df['Month'] = df['Date of Data Collection'].dt.month
	df['Quarter'] = df['Date of Data Collection'].dt.quarter

	# Vehicle age years
	if 'Year' in df.columns:
		df['vehicle_age_years'] = df['Date of Data Collection'].dt.year - pd.to_numeric(df['Year'], errors='coerce')
		df['vehicle_age_years'] = df['vehicle_age_years'].clip(lower=0).fillna(df['vehicle_age_years'].median())

	return df

def derive_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
	# Number of late renewals in last 3 years - prefer existing numeric, else infer from Compliance History
	if 'Number of Late Renewals in Last 3 Years' in df.columns:
		df['Number of Late Renewals in Last 3 Years'] = pd.to_numeric(df['Number of Late Renewals in Last 3 Years'], errors='coerce')
	late_from_history = df['Compliance History'].map({
		'Always Compliant': 0,
		'Occasional Late': 2,
		'Frequently Non-compliant': 4
	})
	df['Number of Late Renewals in Last 3 Years'] = df['Number of Late Renewals in Last 3 Years'].fillna(late_from_history).fillna(0).clip(lower=0).astype(int)

	# Average renewal lag days - keep if present, else approximate using expiration vs last renewal
	df['Average Renewal Lag Days'] = pd.to_numeric(df.get('Average Renewal Lag Days', np.nan), errors='coerce')
	m = df['Average Renewal Lag Days'].isna() & df['Last Renewal Date'].notna() & df['Expiration Date'].notna()
	# Approximate lag as delay beyond expected 365-day cycle; negative -> 0
	df.loc[m, 'Average Renewal Lag Days'] = (
		(df.loc[m, 'Last Renewal Date'] - (df.loc[m, 'Expiration Date'] - pd.to_timedelta(365, unit='D')))
		.dt.days.clip(lower=0)
	)
	df['Average Renewal Lag Days'] = df['Average Renewal Lag Days'].fillna(0).clip(lower=0)

	# Late renewal rate per year
	df['late_renewal_rate'] = df['Number of Late Renewals in Last 3 Years'] / 3.0

	# Renewal frequency proxy (lower avg lag -> higher frequency); normalized 0..1
	max_lag = max(1.0, df['Average Renewal Lag Days'].max() if pd.notna(df['Average Renewal Lag Days'].max()) else 1.0)
	df['renewal_frequency'] = 1.0 - (df['Average Renewal Lag Days'] / max_lag)

	return df

def derive_channel_features(df: pd.DataFrame) -> pd.DataFrame:
	# Agent synchronization lag: zero when not using Agent/In-person; impute median otherwise
	df['Agent Synchronization Lag'] = pd.to_numeric(df.get('Agent Synchronization Lag', np.nan), errors='coerce')
	agent_like = df['Agent Service Used'].astype(str).str.lower().isin(['agent', 'in-person', 'in person'])
	df.loc[~agent_like, 'Agent Synchronization Lag'] = 0
	median_sync = df.loc[agent_like, 'Agent Synchronization Lag'].median()
	df.loc[agent_like & df['Agent Synchronization Lag'].isna(), 'Agent Synchronization Lag'] = median_sync if pd.notna(median_sync) else 0

	# Payment digitalization score
	digital_mapping = {
		'Online': 1.0,
		'Mobile Money': 0.8,
		'Agent': 0.4,
		'In-person': 0.3,
		'Cash': 0.0,
		'Bank Transfer': 0.6
	}
	df['payment_digital_score'] = df['Preferred Payment Mode'].map(digital_mapping).fillna(0.5)

	# Regional risk score
	region_risk = {'Urban': 0.3, 'Peri-urban': 0.5, 'Rural': 0.7}
	df['region_risk_score'] = df['Region'].map(region_risk).fillna(0.5)

	# Agent efficiency (1 - normalized lag)
	max_delay = max(1.0, df['Agent Synchronization Lag'].max() if pd.notna(df['Agent Synchronization Lag'].max()) else 1.0)
	df['agent_efficiency'] = 1.0 - (df['Agent Synchronization Lag'] / max_delay)

	return df

def finalize_targets(df: pd.DataFrame) -> pd.DataFrame:
	# Target boolean
	df['is_licensed'] = df['License Status'].map({'Compliant': True, 'Non-compliant': False, 'Expired': False}).fillna(False).astype(bool)
	return df

def augment_dataset(df: pd.DataFrame, multiplier: int = 3) -> pd.DataFrame:
	"""Augment dataset by generating synthetic variations of existing records."""
	print(f"Augmenting dataset by {multiplier}x...")
	original_count = len(df)
	augmented_rows = []
	
	# Get unique combinations for variety
	vehicle_types = df['Vehicle Type'].dropna().unique()
	regions = df['Region'].dropna().unique()
	payment_modes = df['Preferred Payment Mode'].dropna().unique()
	income_levels = df['Income Level'].dropna().unique()
	
	# Generate synthetic records
	for _ in range(multiplier - 1):
		for idx, row in df.iterrows():
			new_row = row.copy()
			
			# Add controlled noise to numeric columns
			numeric_cols = df.select_dtypes(include=[np.number]).columns
			for col in numeric_cols:
				if col in new_row.index and pd.notna(new_row[col]):
					noise = np.random.normal(0, 0.1 * abs(new_row[col]))
					new_row[col] = max(0, new_row[col] + noise) if new_row[col] >= 0 else new_row[col] + noise
			
			# Vary categorical features slightly
			if len(vehicle_types) > 1 and np.random.random() < 0.2:
				new_row['Vehicle Type'] = np.random.choice(vehicle_types)
			if len(regions) > 1 and np.random.random() < 0.15:
				new_row['Region'] = np.random.choice(regions)
			if len(payment_modes) > 1 and np.random.random() < 0.2:
				new_row['Preferred Payment Mode'] = np.random.choice(payment_modes)
			if len(income_levels) > 1 and np.random.random() < 0.15:
				new_row['Income Level'] = np.random.choice(income_levels)
			
			# Generate new unique IDs
			new_row['Vehicle ID'] = f"V{original_count + len(augmented_rows) + 1:06d}"
			new_row['Owner ID'] = f"O{np.random.randint(1000, 99999)}"
			
			augmented_rows.append(new_row)
	
	augmented_df = pd.DataFrame(augmented_rows)
	result_df = pd.concat([df, augmented_df], ignore_index=True)
	print(f"Dataset augmented: {original_count} -> {len(result_df)} rows")
	return result_df

def main():
	print("Loading raw dataset...")
	df = pd.read_csv(RAW_PATH)
	df = ensure_columns(df)

	print("Normalizing values...")
	df = normalize_values(df)

	print("Deriving temporal features...")
	df = derive_temporal_features(df)

	print("Deriving behavioral features...")
	df = derive_behavioral_features(df)

	print("Deriving channel features...")
	df = derive_channel_features(df)

	print("Finalizing target column...")
	df = finalize_targets(df)

	# Augment dataset to increase size
	if len(df) < 10000:
		df = augment_dataset(df, multiplier=3)

	# Save processed
	os.makedirs(PROCESSED_DIR, exist_ok=True)
	df.to_csv(PROCESSED_PATH, index=False)

	print(f"Enrichment complete. Rows: {len(df)}, Columns: {len(df.columns)}")
	print(f"Saved to: {PROCESSED_PATH}")

if __name__ == "__main__":
	main()

