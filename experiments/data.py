import numpy as np
import matplotlib.pyplot as plt

def generate_circle_data(n_samples=1000, x_range=(-5, 5), y_range=(-5, 5)):
    x_coords = np.random.uniform(x_range[0], x_range[1], n_samples)
    y_coords = np.random.uniform(y_range[0], y_range[1], n_samples)
    
    inputs = np.column_stack([x_coords, y_coords])
    
    outputs = np.sqrt(x_coords**2 + y_coords**2)
    
    return inputs, outputs

def save_data(inputs, outputs, filename_prefix="circle_data"):
    np.save(f"{filename_prefix}_inputs.npy", inputs)
    np.save(f"{filename_prefix}_outputs.npy", outputs)
    
    np.savetxt(f"{filename_prefix}_inputs.txt", inputs, 
               header="x_coordinate y_coordinate", 
               fmt="%.6f")
    np.savetxt(f"{filename_prefix}_outputs.txt", outputs, 
               header="distance_from_origin", 
               fmt="%.6f")
    
    print(f"Data saved as:")
    print(f"  - {filename_prefix}_inputs.npy/.txt")
    print(f"  - {filename_prefix}_outputs.npy/.txt")

def load_data(filename_prefix="circle_data"):
    """
    Load previously saved data
    
    Returns:
        inputs: input coordinates array
        outputs: distance values array
    """
    inputs = np.load(f"{filename_prefix}_inputs.npy")
    outputs = np.load(f"{filename_prefix}_outputs.npy")
    
    print(f"Loaded data:")
    print(f"  - {inputs.shape[0]} data points")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Output shape: {outputs.shape}")
    
    return inputs, outputs

def visualize_data(inputs, outputs, sample_size=500):
    """
    Create visualizations to understand the generated data
    
    Args:
        inputs: input coordinates
        outputs: distance values
        sample_size: how many points to plot (for readability)
    """
    
    # Take a random sample for plotting (if dataset is large)
    if len(inputs) > sample_size:
        indices = np.random.choice(len(inputs), sample_size, replace=False)
        plot_inputs = inputs[indices]
        plot_outputs = outputs[indices]
    else:
        plot_inputs = inputs
        plot_outputs = outputs
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Scatter plot colored by distance
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(plot_inputs[:, 0], plot_inputs[:, 1], 
                         c=plot_outputs, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Distance from origin')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Data Points\n(Color = Distance from origin)')
    plt.grid(True, alpha=0.3)
    
    # Add circles for reference
    circle_radii = [1, 2, 3, 4]
    for radius in circle_radii:
        circle = plt.Circle((0, 0), radius, fill=False, color='red', alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.axis('equal')
    
    # Plot 2: Distribution of distances
    plt.subplot(1, 3, 2)
    plt.hist(plot_outputs, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance from origin')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distance vs X coordinate (to see the relationship)
    plt.subplot(1, 3, 3)
    plt.scatter(plot_inputs[:, 0], plot_outputs, alpha=0.5, s=10)
    plt.xlabel('X coordinate')
    plt.ylabel('Distance from origin')
    plt.title('Distance vs X coordinate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def examine_data(inputs, outputs):
    """
    Print some statistics and examples to understand the data
    """
    print("=== DATA EXAMINATION ===")
    print(f"Number of data points: {len(inputs)}")
    print(f"Input range - X: [{inputs[:, 0].min():.2f}, {inputs[:, 0].max():.2f}]")
    print(f"Input range - Y: [{inputs[:, 1].min():.2f}, {inputs[:, 1].max():.2f}]")
    print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    print(f"Average distance: {outputs.mean():.2f}")
    
    print("\nFirst 10 data points:")
    print("   X      Y    |  Distance")
    print("-" * 25)
    for i in range(10):
        x, y = inputs[i]
        dist = outputs[i]
        print(f"{x:6.2f} {y:6.2f} | {dist:6.2f}")
    
    # Verify our calculation
    print(f"\nVerification (manual calculation of first point):")
    x, y = inputs[0]
    manual_dist = np.sqrt(x**2 + y**2)
    print(f"Point: ({x:.2f}, {y:.2f})")
    print(f"Stored distance: {outputs[0]:.6f}")
    print(f"Manual calculation: √({x:.2f}² + {y:.2f}²) = {manual_dist:.6f}")
    print(f"Match: {np.isclose(outputs[0], manual_dist)}")

if __name__ == "__main__":
    print("=== CIRCLE DATA GENERATION ===\n")
    
    # Step 1: Generate the data
    print("1. Generating data...")
    n_points = 1000
    inputs, outputs = generate_circle_data(n_samples=n_points, 
                                         x_range=(-5, 5), 
                                         y_range=(-5, 5))
    
    # Step 2: Examine the data
    print("\n2. Examining generated data...")
    examine_data(inputs, outputs)
    
    # Step 3: Save the data
    print("\n3. Saving data to files...")
    save_data(inputs, outputs, "circle_training_data")
    
    # Step 4: Visualize the data
    print("\n4. Creating visualizations...")
    visualize_data(inputs, outputs)
    
    # Step 5: Test loading the data
    print("\n5. Testing data loading...")
    loaded_inputs, loaded_outputs = load_data("circle_training_data")
    
    # Verify loaded data matches original
    print(f"Data integrity check: {np.allclose(inputs, loaded_inputs) and np.allclose(outputs, loaded_outputs)}")
    
    print("\n=== READY FOR MACHINE LEARNING ===")
    print("Data files created successfully!")
    print("Next step: Use this data to train polynomial regression model")