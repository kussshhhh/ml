import numpy as np
import matplotlib.pyplot as plt
import json

class PolynomialRegression:
    def __init__(self):
        self.weights = None
        self.feature_names = ['constant', 'x', 'y', 'x²', 'y²', 'xy']
        self.trained = False
    
    def create_polynomial_features(self, X):
        """
        Convert (x, y) coordinates to polynomial features
        Input: X shape (n_samples, 2) - [x, y] coordinates
        Output: Features shape (n_samples, 6) - [1, x, y, x², y², xy]
        """
        n_samples = X.shape[0]
        x, y = X[:, 0], X[:, 1]
        
        # Create polynomial features matrix
        features = np.column_stack([
            np.ones(n_samples),  # constant term (bias)
            x,                   # linear x
            y,                   # linear y  
            x**2,               # quadratic x
            y**2,               # quadratic y
            x*y                 # interaction term x*y
        ])
        return features
    
    def fit(self, X, y):
        """
        Train the polynomial regression model
        X: input coordinates (n_samples, 2) - [x, y] pairs
        y: target distances (n_samples,) - distance from origin
        """
        print("Training polynomial regression...")
        
        # Step 1: Create polynomial features
        features = self.create_polynomial_features(X)
        print(f"Created feature matrix: {features.shape}")
        
        # Step 2: Solve normal equation: weights = (X^T * X)^(-1) * X^T * y
        print("Solving normal equation...")
        XtX = features.T @ features  # X transpose times X
        Xty = features.T @ y         # X transpose times y
        
        # Solve the linear system (more stable than computing inverse)
        self.weights = np.linalg.solve(XtX, Xty)
        self.trained = True
        
        print("Training completed!")
        self.print_learned_equation()
    
    def predict(self, X):
        """Make predictions on new data"""
        if not self.trained:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        features = self.create_polynomial_features(X)
        predictions = features @ self.weights
        return predictions
    
    def print_learned_equation(self):
        """Print the learned equation in readable format"""
        print("\n=== LEARNED POLYNOMIAL EQUATION ===")
        print("distance ≈ ", end="")
        
        equation_parts = []
        for i, (name, weight) in enumerate(zip(self.feature_names, self.weights)):
            if abs(weight) > 0.0001:  # Only show significant terms
                if i == 0:  # constant term
                    equation_parts.append(f"{weight:.4f}")
                else:
                    sign = "+" if weight >= 0 else "-"
                    if len(equation_parts) == 0:  # first term
                        equation_parts.append(f"{weight:.4f}*{name}")
                    else:
                        equation_parts.append(f" {sign} {abs(weight):.4f}*{name}")
        
        print("".join(equation_parts))
        
        print("\nWeight breakdown:")
        for name, weight in zip(self.feature_names, self.weights):
            print(f"  {name:8s}: {weight:8.4f}")
        
        print(f"\nTrue equation: distance = √(x² + y²)")
    
    def save_weights(self, filename="polynomial_weights"):
        """Save the learned weights to files"""
        if not self.trained:
            raise ValueError("Model not trained yet! Nothing to save.")
        
        # Save as numpy array (binary)
        np.save(f"{filename}.npy", self.weights)
        
        # Save as JSON (human readable)
        weight_dict = {
            name: float(weight) 
            for name, weight in zip(self.feature_names, self.weights)
        }
        weight_dict['trained'] = True
        weight_dict['equation_type'] = 'polynomial_regression'
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(weight_dict, f, indent=2)
        
        # Save as text file (most readable)
        with open(f"{filename}.txt", 'w') as f:
            f.write("POLYNOMIAL REGRESSION WEIGHTS\n")
            f.write("=" * 30 + "\n")
            f.write(f"Training completed: {self.trained}\n")
            f.write(f"Number of features: {len(self.weights)}\n\n")
            
            f.write("LEARNED EQUATION:\n")
            f.write("distance ≈ ")
            equation_parts = []
            for i, (name, weight) in enumerate(zip(self.feature_names, self.weights)):
                if abs(weight) > 0.0001:
                    if i == 0:
                        equation_parts.append(f"{weight:.4f}")
                    else:
                        sign = "+" if weight >= 0 else "-"
                        if len(equation_parts) == 0:
                            equation_parts.append(f"{weight:.4f}*{name}")
                        else:
                            equation_parts.append(f" {sign} {abs(weight):.4f}*{name}")
            f.write("".join(equation_parts) + "\n\n")
            
            f.write("WEIGHT BREAKDOWN:\n")
            for name, weight in zip(self.feature_names, self.weights):
                f.write(f"{name:12s}: {weight:10.6f}\n")
            
            f.write(f"\nTRUE EQUATION:\n")
            f.write(f"distance = √(x² + y²)\n")
        
        print(f"\nWeights saved as:")
        print(f"  - {filename}.npy (binary)")
        print(f"  - {filename}.json (JSON)")
        print(f"  - {filename}.txt (readable)")
    
    def load_weights(self, filename="polynomial_weights"):
        """Load previously saved weights"""
        try:
            self.weights = np.load(f"{filename}.npy")
            self.trained = True
            print(f"Weights loaded from {filename}.npy")
            self.print_learned_equation()
        except FileNotFoundError:
            print(f"Weight file {filename}.npy not found!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.trained:
            raise ValueError("Model not trained yet!")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test)**2)
        mae = np.mean(np.abs(predictions - y_test))
        
        # R-squared
        ss_res = np.sum((y_test - predictions)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\n=== MODEL EVALUATION ===")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R-squared Score: {r2:.6f}")
        
        return mse, mae, r2

def load_training_data(filename_prefix="circle_training_data"):
    """Load the data we generated earlier"""
    try:
        inputs = np.load(f"{filename_prefix}_inputs.npy")
        outputs = np.load(f"{filename_prefix}_outputs.npy")
        print(f"Loaded training data: {inputs.shape[0]} samples")
        return inputs, outputs
    except FileNotFoundError:
        print(f"Training data files not found!")
        print(f"Run the data generation script first.")
        return None, None

def test_specific_points(model):
    """Test the model on some specific points for intuition"""
    print(f"\n=== TESTING ON SPECIFIC POINTS ===")
    
    test_points = np.array([
        [0, 0],    # origin
        [1, 0],    # unit x
        [0, 1],    # unit y
        [1, 1],    # diagonal
        [3, 4],    # 3-4-5 triangle
        [-2, -2],  # negative quadrant
        [0, 5],    # far on y-axis
    ])
    
    print("Point        | True Distance | Predicted | Error")
    print("-" * 50)
    
    for point in test_points:
        true_dist = np.sqrt(point[0]**2 + point[1]**2)
        pred_dist = model.predict(point.reshape(1, -1))[0]
        error = abs(pred_dist - true_dist)
        
        print(f"({point[0]:2.0f}, {point[1]:2.0f})      | {true_dist:8.3f}      | {pred_dist:8.3f}  | {error:.3f}")

if __name__ == "__main__":
    print("=== POLYNOMIAL REGRESSION TRAINING ===\n")
    
    # Step 1: Load the data
    print("1. Loading training data...")
    X_train, y_train = load_training_data()
    
    if X_train is None:
        print("Please run the data generation script first!")
        exit()
    
    # Step 2: Create and train the model
    print("\n2. Creating polynomial regression model...")
    model = PolynomialRegression()
    
    print("\n3. Training the model...")
    model.fit(X_train, y_train)
    
    # Step 3: Save the weights
    print("\n4. Saving learned weights...")
    model.save_weights("polynomial_circle_weights")
    
    # Step 4: Evaluate on training data
    print("\n5. Evaluating model...")
    model.evaluate(X_train, y_train)
    
    # Step 5: Test on specific points
    test_specific_points(model)
    
    # Step 6: Generate some test data and evaluate
    print("\n6. Testing on new data...")
    # Generate fresh test data
    from data import generate_circle_data
    X_test, y_test = generate_circle_data(n_samples=200, x_range=(-5, 5), y_range=(-5, 5))
    model.evaluate(X_test, y_test)
    
    print("\n=== TRAINING COMPLETE ===")
    print("Check the weight files to see what the model learned!")
