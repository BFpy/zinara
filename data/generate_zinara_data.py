import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define constants for Zimbabwe-specific data
ZIMBABWE_CITIES = [
    'Harare', 'Bulawayo', 'Chitungwiza', 'Mutare', 'Gweru', 'Kwekwe', 'Kadoma',
    'Masvingo', 'Chinhoyi', 'Norton', 'Marondera', 'Chegutu', 'Zvishavane',
    'Bindura', 'Beitbridge', 'Victoria Falls', 'Hwange', 'Redcliff', 'Ruwa',
    'Kariba', 'Chiredzi', 'Plumtree', 'Gwanda', 'Shurugwi', 'Lupane'
]

ZIMBABWE_PROVINCES = [
    'Harare', 'Bulawayo', 'Manicaland', 'Mashonaland Central', 'Mashonaland East',
    'Mashonaland West', 'Masvingo', 'Matabeleland North', 'Matabeleland South',
    'Midlands'
]

VEHICLE_MAKES = [
    'Toyota', 'Ford', 'Nissan', 'Mazda', 'Honda', 'Mitsubishi', 'Isuzu',
    'Mercedes-Benz', 'Volkswagen', 'BMW', 'Audi', 'Land Rover', 'Hyundai',
    'Kia', 'Chevrolet', 'Foton', 'JAC', 'Great Wall', 'Haval'
]

VEHICLE_MODELS = {
    'Toyota': ['Corolla', 'Hilux', 'Fortuner', 'Land Cruiser', 'Camry', 'RAV4', 'Prado', 'Yaris', 'Avanza'],
    'Ford': ['Ranger', 'Everest', 'F-150', 'Explorer', 'Focus', 'Fiesta'],
    'Nissan': ['Navara', 'Hardbody', 'Sunny', 'Almera', 'Patrol', 'X-Trail'],
    'Mazda': ['BT-50', 'CX-5', 'Demio', '323'],
    'Honda': ['CR-V', 'Civic', 'Accord', 'Fit'],
    'Mitsubishi': ['L200', 'Pajero', 'Outlander', 'Triton'],
    'Isuzu': ['KB', 'D-Max', 'MU-X'],
    'Mercedes-Benz': ['Sprinter', 'Actros', 'C-Class', 'E-Class'],
    'Volkswagen': ['Polo', 'Golf', 'Tiguan', 'Amarok'],
    'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
    'Audi': ['A3', 'A4', 'Q5', 'Q7'],
    'Land Rover': ['Defender', 'Discovery', 'Range Rover'],
    'Hyundai': ['Tucson', 'Santa Fe', 'i10', 'i20'],
    'Kia': ['Sportage', 'Sorento', 'Rio', 'Picanto'],
    'Chevrolet': ['Spark', 'Cruze', 'Trailblazer'],
    'Foton': ['Tunland', 'View', 'Sauvana'],
    'JAC': ['T6', 'S2', 'M1'],
    'Great Wall': ['Wingle', 'Hover', 'Steed'],
    'Haval': ['H6', 'H9', 'Jolion']
}

PAYMENT_MODES = ['Online', 'Mobile Money', 'Agent', 'Cash', 'Bank Transfer']

ONLINE_PLATFORMS = ['Zinara Portal', 'Paynow', 'Ecocash', 'OneMoney', 'Telecash', 'Innbucks']

AGENT_SERVICES = ['In-person', 'Agent', 'Online']

INCOME_LEVELS = ['Low', 'Middle', 'High', 'Very High']

VEHICLE_TYPES = ['Sedan', 'SUV', 'Truck', 'Pickup', 'Hatchback', 'Minivan', 'Motorcycle']

REGIONS = ['Urban', 'Peri-urban', 'Rural']

def generate_vehicle_data(num_records=3000):
    data = []

    for i in range(num_records):
        # Generate unique IDs
        vehicle_id = f"VH-{str(uuid.uuid4())[:8].upper()}"
        owner_id = f"OWN-{str(uuid.uuid4())[:8].upper()}"

        # Vehicle details
        make = random.choice(VEHICLE_MAKES)
        model = random.choice(VEHICLE_MODELS[make])
        year = random.randint(2000, 2024)
        vehicle_type = random.choice(VEHICLE_TYPES)

        # Location
        city = random.choice(ZIMBABWE_CITIES)
        province = random.choice(ZIMBABWE_PROVINCES)
        region = random.choice(REGIONS)

        # License status and dates
        license_status = random.choices(['Compliant', 'Non-compliant', 'Expired'], weights=[0.7, 0.2, 0.1])[0]

        # Generate dates
        data_collection_date = datetime.now() - timedelta(days=random.randint(0, 365*2))
        last_renewal_date = data_collection_date - timedelta(days=random.randint(30, 365*2))

        if license_status == 'Compliant':
            expiration_date = last_renewal_date + timedelta(days=365)
        elif license_status == 'Non-compliant':
            expiration_date = last_renewal_date + timedelta(days=random.randint(-30, 364))
        else:
            expiration_date = last_renewal_date - timedelta(days=random.randint(1, 365))

        # Fines and violations
        fine_amount = random.choices([0, random.uniform(50, 5000)], weights=[0.8, 0.2])[0]
        previous_violations = random.choices([0, 1, 2, 3, 4, 5], weights=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])[0]

        # Service details
        agent_service_used = random.choice(AGENT_SERVICES)
        online_platform_used = random.choice(ONLINE_PLATFORMS) if agent_service_used == 'Online' else None
        agent_hours_used = random.uniform(0.5, 8) if agent_service_used in ['In-person', 'Agent'] else 0

        # Owner details
        income_level = random.choice(INCOME_LEVELS)
        num_vehicles_owned = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.15, 0.07, 0.03])[0]

        # Compliance history
        compliance_history = random.choices(['Always Compliant', 'Occasional Late', 'Frequently Non-compliant'],
                                          weights=[0.6, 0.3, 0.1])[0]

        # User feedback and predictive score
        user_feedback_score = random.uniform(1, 5) if random.random() > 0.3 else None
        predictive_score = random.uniform(0, 1)

        # Payment mode
        preferred_payment_mode = random.choice(PAYMENT_MODES)

        # Feature engineering columns
        days_since_last_renewal = (data_collection_date - last_renewal_date).days
        num_late_renewals_3years = random.randint(0, 5)
        avg_renewal_lag_days = random.uniform(0, 30) if num_late_renewals_3years > 0 else 0
        agent_sync_lag = random.uniform(0, 24) if agent_service_used in ['In-person', 'Agent'] else 0

        # Time of year
        month = data_collection_date.month
        quarter = (month - 1) // 3 + 1

        record = {
            'Vehicle ID': vehicle_id,
            'Owner ID': owner_id,
            'Make': make,
            'Model': model,
            'Year': year,
            'Vehicle Type': vehicle_type,
            'License Status': license_status,
            'Last Renewal Date': last_renewal_date.strftime('%Y-%m-%d'),
            'Expiration Date': expiration_date.strftime('%Y-%m-%d'),
            'Fine Amount': round(fine_amount, 2) if fine_amount > 0 else 0,
            'Agent Service Used': agent_service_used,
            'Online Platform Used': online_platform_used,
            'Compliance History': compliance_history,
            'Geographic Location': f"{city}, {province}",
            'Region': region,
            'Income Level': income_level,
            'Number of Vehicles Owned': num_vehicles_owned,
            'Previous Violations': previous_violations,
            'Agent Hours Used': round(agent_hours_used, 2),
            'User Feedback Score': round(user_feedback_score, 1) if user_feedback_score else None,
            'Predictive Score': round(predictive_score, 3),
            'Preferred Payment Mode': preferred_payment_mode,
            'Date of Data Collection': data_collection_date.strftime('%Y-%m-%d'),
            'Days Since Last Renewal': days_since_last_renewal,
            'Number of Late Renewals in Last 3 Years': num_late_renewals_3years,
            'Average Renewal Lag Days': round(avg_renewal_lag_days, 2),
            'Agent Synchronization Lag': round(agent_sync_lag, 2),
            'Month': month,
            'Quarter': quarter
        }

        data.append(record)

    return pd.DataFrame(data)

def add_data_uncleanliness(df):
    """Add realistic data quality issues"""
    # Randomly make some values missing
    for col in ['Fine Amount', 'User Feedback Score', 'Agent Hours Used', 'Income Level']:
        mask = np.random.random(len(df)) < 0.1  # 10% missing
        df.loc[mask, col] = None

    # Add some inconsistent date formats
    date_cols = ['Last Renewal Date', 'Expiration Date', 'Date of Data Collection']
    for col in date_cols:
        mask = np.random.random(len(df)) < 0.05  # 5% inconsistent
        df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x.replace('-', '/'))

    # Add some duplicate vehicle IDs (but not too many)
    duplicate_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    if len(duplicate_indices) > 0:
        duplicate_vehicle_id = df.iloc[duplicate_indices[0]]['Vehicle ID']
        df.loc[duplicate_indices, 'Vehicle ID'] = duplicate_vehicle_id

    # Add some outliers in numeric columns
    numeric_cols = ['Fine Amount', 'Agent Hours Used', 'Days Since Last Renewal']
    for col in numeric_cols:
        mask = np.random.random(len(df)) < 0.03  # 3% outliers
        df.loc[mask, col] = df.loc[mask, col] * 10  # Make them 10x larger

    return df

if __name__ == "__main__":
    print("Generating Zimbabwe vehicle licensing dataset...")

    # Generate the data
    df = generate_vehicle_data(3000)

    # Add uncleanliness
    df = add_data_uncleanliness(df)

    # Save to CSV
    output_file = "zinara_vehicle_licensing_data_2015_2024.csv"
    df.to_csv(output_file, index=False)

    print(f"Dataset generated with {len(df)} records")
    print(f"Saved to {output_file}")
    print("\nSample of generated data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values summary:")
    print(df.isnull().sum())