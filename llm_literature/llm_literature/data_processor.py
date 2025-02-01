import pandas as pd

class DataProcessor:
    def __init__(self, data_source):
        """
        Base class for processing data.

        Parameters:
        - data_source (str): Path to the data source file.
        """
        self.data_source = data_source

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method!")


class CSVProcessor(DataProcessor):
    def load_data(self):
        """
        Load and preprocess data from a CSV file.

        Returns:
        - data (pd.DataFrame): Preprocessed DataFrame.
        """
        data = pd.read_csv(self.data_source)
        required_columns = [
            "Author", "Author full names", "Author ID", "Title", "DOI", 
            "Year", "abstract", "Source", "reference", "Author with Institution"
        ]
        # Ensure all required columns are present
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Select required columns
        data = data[required_columns]
        # Drop rows with any missing required data
        data = data.dropna(subset=required_columns)
        # Exclude articles without abstracts
        data = data[data["abstract"] != "[No abstract available]"]
        return data