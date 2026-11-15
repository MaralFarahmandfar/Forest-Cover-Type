import pandas as pd
from pathlib import Path
import numpy as np

# Project root = parent of the src/ folder, so paths work from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT/"data"/"covtype.data.gz"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "covtype_processed.npz"

numeric_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

wilderness_cols = [f"Wilderness_Area_{i}" for i in range(1, 5)]
soil_cols = [f"Soil_Type_{i}" for i in range(1, 41)]
target_col = "Cover_Type"
 
col_names = numeric_cols + wilderness_cols + soil_cols + [target_col]

def load_covertype(path=DATA_PATH):
    """Load the raw Covertype file and return a DataFrame with 55 columns
    (54 features + the Cover_Type target, classes 1 to 7)."""
    return pd.read_csv(path, header=None, names=col_names)


def load_processed(path=PROCESSED_PATH):
    """Load the preprocessed arrays saved by notebook 02.
    Returns a dict with X_train, X_dev, X_test, y_train, y_dev, y_test."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
