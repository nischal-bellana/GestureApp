import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys



def train_static_gesture_model(mode="static"):
    csv_file='Training_Samples/gesture_data.csv'
    model_save_path='static_gesture_model.pkl' 
    RANDOM_STATE = 42

    if mode=='dynamic':
        RANDOM_STATE = 420
        model_save_path = 'dynamic_gesture_model.pkl'
        csv_file = 'Training_Samples/gesture_dynamic_data.csv'
    
    print(f"Loading dataset from {csv_file}...")
    # 1. Load the Data
    # Assuming no header row since the previous script used csv.writer directly
    df = pd.read_csv(csv_file, header=None)
    
    # The first column (index 0) contains the labels, the rest (1 to 42) are features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    
    print(f"Total samples loaded: {len(X)}")
    print(f"Classes found: {np.unique(y)}")
    
    # 2. Split into Training and Testing sets
    # We use 20% of the data to test how well the model learned
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    print("\nTraining the Random Forest model...")
    
    # 3. Initialize and Train the Model
    # n_estimators=100 means 100 decision trees. It's a good default balance of speed and accuracy.
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    print("\nEvaluating model performance on test data:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save the Model
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nSuccess! Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    train_static_gesture_model(*sys.argv[1:])
