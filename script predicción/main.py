
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline import Pipeline
from dataloader import DataProcessor, TMP_PATH, LOGS_PATH
from model import RANDOM_SEED
from codecarbon import EmissionsTracker
import warnings
warnings.filterwarnings('ignore')
import time
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Ensemble', 
                        help='Model to train and predict with')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to run the prediction phase')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set random seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f'Seed set to {RANDOM_SEED}')
    
    # Create necessary directories
    for path in [TMP_PATH, LOGS_PATH]:
        path.mkdir(exist_ok=True)
    
    # Load data
    X, y = DataProcessor.load_training_data()
    
    # Initialize and run pipeline
    pipeline = Pipeline(args.model)
    _, results = pipeline.train(X, y)
    
    # Save results
    results_df = pd.DataFrame([results])
    csv_filename = LOGS_PATH / 'results.csv'
    
    if not csv_filename.exists():
        results_df.to_csv(csv_filename, index=False)
    else:
        results_df.to_csv(csv_filename, mode='a', header=False, index=False)
    if args.predict:
        # Prediction phase
        print("\nStarting prediction phase...")
        predictions = pipeline.predict(X, y, team_name='HAL9000')
        print("Predictions completed and saved.")
    else:
        print("Training phase completed.")
if __name__ == '__main__':
    emissions_tracker = EmissionsTracker()
    emissions_tracker.start()
    start = time.time()
    main()
    end = time.time()
    emissions = emissions_tracker.stop()
    # The emissions object contains the CO2 emissions in kg , we parse it to grams
    emissions  = emissions * 1000
    print(f"Total CO2 emissions: {emissions} grams")
    print(f"Total execution time: {end - start} seconds")