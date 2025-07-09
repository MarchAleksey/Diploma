import os
import uuid
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn import datasets

from backend.core.config import settings

class DatasetService:
    """Service for handling datasets."""
    
    def __init__(self):
        """Initialize the dataset service."""
        self.data_dir = settings.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        # Generate built-in datasets if the directory is empty
        if not os.listdir(self.data_dir):
            print("Data directory is empty. Generating built-in datasets...")
            self.generate_builtin_datasets()
        else:
            print(f"Found {len(os.listdir(self.data_dir))/2} datasets in {self.data_dir}")
    
    def generate_builtin_datasets(self):
        """Generate built-in datasets from scikit-learn."""
        try:
            # 1. Generate Iris dataset
            print("Generating Iris dataset...")
            iris = datasets.load_iris()
            iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                columns=iris['feature_names'] + ['target'])
            iris_df['target'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            
            # Save Iris dataset
            iris_path = os.path.join(self.data_dir, "iris.csv")
            iris_df.to_csv(iris_path, index=False)
            
            # Save Iris metadata
            iris_meta = {
                "original_filename": "iris.csv",
                "target_column": "target",
                "description": "Iris flower dataset for classification"
            }
            with open(os.path.join(self.data_dir, "iris.json"), 'w') as f:
                json.dump(iris_meta, f)
            
            # 2. Generate Boston dataset
            print("Generating Boston Housing dataset...")
            try:
                boston = datasets.load_boston()
                boston_df = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
                boston_df['MEDV'] = boston['target']
                
                # Save Boston dataset
                boston_path = os.path.join(self.data_dir, "boston.csv")
                boston_df.to_csv(boston_path, index=False)
                
                # Save Boston metadata
                boston_meta = {
                    "original_filename": "boston.csv",
                    "target_column": "MEDV",
                    "description": "Boston housing dataset for regression"
                }
                with open(os.path.join(self.data_dir, "boston.json"), 'w') as f:
                    json.dump(boston_meta, f)
            except:
                # Boston dataset might be deprecated in newer scikit-learn versions
                print("Boston dataset not available, skipping...")
            
            # 3. Generate Breast Cancer dataset
            print("Generating Breast Cancer dataset...")
            cancer = datasets.load_breast_cancer()
            cancer_df = pd.DataFrame(data=np.c_[cancer['data'], cancer['target']],
                                    columns=cancer['feature_names'] + ['target'])
            cancer_df['target'] = cancer_df['target'].map({0: 'malignant', 1: 'benign'})
            
            # Save Breast Cancer dataset
            cancer_path = os.path.join(self.data_dir, "breast_cancer.csv")
            cancer_df.to_csv(cancer_path, index=False)
            
            # Save Breast Cancer metadata
            cancer_meta = {
                "original_filename": "breast_cancer.csv",
                "target_column": "target",
                "description": "Breast cancer Wisconsin dataset for binary classification"
            }
            with open(os.path.join(self.data_dir, "breast_cancer.json"), 'w') as f:
                json.dump(cancer_meta, f)
            
            # 4. Generate MNIST subset (using only a small portion to keep it manageable)
            print("Generating MNIST subset dataset...")
            mnist = datasets.load_digits()  # This is a smaller version of MNIST with 8x8 images
            # Flatten the images and create a DataFrame
            n_samples = len(mnist.images)
            data = mnist.images.reshape((n_samples, -1))
            
            # Create column names for each pixel
            pixel_columns = [f'pixel_{i}' for i in range(data.shape[1])]
            
            mnist_df = pd.DataFrame(data=data, columns=pixel_columns)
            mnist_df['target'] = mnist.target
            
            # Save MNIST subset dataset
            mnist_path = os.path.join(self.data_dir, "mnist_subset.csv")
            mnist_df.to_csv(mnist_path, index=False)
            
            # Save MNIST subset metadata
            mnist_meta = {
                "original_filename": "mnist_subset.csv",
                "target_column": "target",
                "description": "MNIST digits dataset subset (8x8 images) for multi-class classification"
            }
            with open(os.path.join(self.data_dir, "mnist_subset.json"), 'w') as f:
                json.dump(mnist_meta, f)
            
            # 5. Generate synthetic Titanic dataset (since it's not in scikit-learn)
            print("Generating synthetic Titanic dataset...")
            # Create a synthetic dataset inspired by Titanic
            np.random.seed(42)
            n_samples = 891  # Similar to the original Titanic dataset
            
            # Generate synthetic data
            pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.5])
            sex = np.random.choice(['male', 'female'], size=n_samples, p=[0.65, 0.35])
            age = np.random.normal(30, 14, size=n_samples)
            age = np.clip(age, 0.5, 80)  # Clip ages to reasonable range
            
            # More realistic age distribution
            age[np.random.choice(n_samples, size=int(n_samples*0.05))] = np.random.uniform(0.5, 5, size=int(n_samples*0.05))  # Infants
            
            sibsp = np.random.choice(range(9), size=n_samples, p=[0.65, 0.25, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.01])
            parch = np.random.choice(range(7), size=n_samples, p=[0.65, 0.2, 0.1, 0.03, 0.01, 0.005, 0.005])
            
            # Generate fare based on class
            fare = np.zeros(n_samples)
            fare[pclass == 1] = np.random.normal(100, 50, size=sum(pclass == 1))
            fare[pclass == 2] = np.random.normal(30, 15, size=sum(pclass == 2))
            fare[pclass == 3] = np.random.normal(15, 10, size=sum(pclass == 3))
            fare = np.clip(fare, 0, 500)  # Clip fares to reasonable range
            
            # Generate embarked port
            embarked = np.random.choice(['C', 'Q', 'S'], size=n_samples, p=[0.2, 0.1, 0.7])
            
            # Generate survival based on features (women and children first, higher class better survival)
            survival_prob = np.zeros(n_samples)
            survival_prob += (sex == 'female') * 0.5  # Women have higher survival rate
            survival_prob += (age < 12) * 0.3  # Children have higher survival rate
            survival_prob += (pclass == 1) * 0.3  # First class has higher survival rate
            survival_prob += (pclass == 2) * 0.1  # Second class has moderate survival rate
            survival_prob = np.clip(survival_prob, 0.1, 0.9)  # Clip probabilities
            
            survived = np.random.binomial(1, survival_prob)
            
            # Create DataFrame
            titanic_df = pd.DataFrame({
                'Survived': survived,
                'Pclass': pclass,
                'Sex': sex,
                'Age': age.round(1),
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare.round(2),
                'Embarked': embarked
            })
            
            # Save Titanic dataset
            titanic_path = os.path.join(self.data_dir, "titanic.csv")
            titanic_df.to_csv(titanic_path, index=False)
            
            # Save Titanic metadata
            titanic_meta = {
                "original_filename": "titanic.csv",
                "target_column": "Survived",
                "description": "Synthetic Titanic dataset for binary classification (survival prediction)"
            }
            with open(os.path.join(self.data_dir, "titanic.json"), 'w') as f:
                json.dump(titanic_meta, f)
            
            print("All datasets generated successfully!")
            return True
        except Exception as e:
            print(f"Error generating built-in datasets: {str(e)}")
            return False

    def get_available_datasets(self) -> List[str]:
        """Get available datasets.
        
        Returns:
            List[str]: List of available datasets.
        """
        # Get all files in the data directory
        files = os.listdir(self.data_dir)
        
        # Filter out non-CSV files
        datasets = [f.split('.')[0] for f in files if f.endswith('.csv')]
        
        return datasets
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset information.
        
        Args:
            dataset_id (str): The dataset ID.
        
        Returns:
            Dict[str, Any]: Dataset information.
        
        Raises:
            FileNotFoundError: If the dataset is not found.
        """
        # Check if dataset exists
        dataset_path = os.path.join(self.data_dir, f"{dataset_id}.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Get metadata
        metadata_path = os.path.join(self.data_dir, f"{dataset_id}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Get target column
        target_column = metadata.get('target_column', df.columns[-1])
        
        # Get features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Determine target type
        if pd.api.types.is_numeric_dtype(y):
            if len(y.unique()) < 10:  # Arbitrary threshold
                target_type = "classification"
            else:
                target_type = "regression"
        else:
            target_type = "classification"
        
        # Return dataset information
        return {
            'name': dataset_id,
            'description': metadata.get('description', ''),
            'n_samples': len(df),
            'n_features': len(X.columns),
            'target_type': target_type,
            'target_column': target_column,
            'preview': df.head().to_dict(orient='records')
        }
    
    def save_dataset(self, content: bytes, filename: str, target_column: str) -> str:
        """Save a dataset.
        
        Args:
            content (bytes): The dataset content.
            filename (str): The dataset filename.
            target_column (str): The target column.
        
        Returns:
            str: The dataset ID.
        """
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Save dataset
        dataset_path = os.path.join(self.data_dir, f"{dataset_id}.csv")
        with open(dataset_path, 'wb') as f:
            f.write(content)
        
        # Save metadata
        metadata = {
            'original_filename': filename,
            'target_column': target_column
        }
        
        metadata_path = os.path.join(self.data_dir, f"{dataset_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return dataset_id
    
    def load_dataset(self, dataset_id: str) -> tuple:
        """Load a dataset.
        
        Args:
            dataset_id (str): The dataset ID.
        
        Returns:
            tuple: Features and target.
        
        Raises:
            FileNotFoundError: If the dataset is not found.
        """
        # Check if dataset exists
        dataset_path = os.path.join(self.data_dir, f"{dataset_id}.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Get metadata
        metadata_path = os.path.join(self.data_dir, f"{dataset_id}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Get target column
        target_column = metadata.get('target_column', df.columns[-1])
        
        # Get features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return X, y, X.columns.tolist()
