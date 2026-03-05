from pathlib import Path
from src.pie_interface.pie_data import PIE

DEFAULT_DATA_PATH = Path("data/PIE_dataset")

def load_pie(data_path: Path | str = DEFAULT_DATA_PATH, regen: bool = False):
    """
    Loads PIE and returns (pie, db). Uses cached pickle unless regen=True.
    """
    pie = PIE(regen_database=regen, data_path=str(data_path))
    db = pie.generate_database()
    return pie, db