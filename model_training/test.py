import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def load_data():
    """Load and preprocess MNIST test data"""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_test, y_test

def evaluate_model(model, x_test, y_test):
    """Run full evaluation on test set"""
    # Basic evaluation
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    inference_time = time.time() - start_time
    
    # Detailed predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'inference_time': inference_time,
        'report': classification_report(y_test, predicted_classes),
        'confusion_matrix': confusion_matrix(y_test, predicted_classes),
        'misclassified': np.where(predicted_classes != y_test)[0]
    }

def save_misclassified_examples(x_test, y_test, predictions, indices, num_examples=10):
    """Save sample misclassified images"""
    predicted_classes = np.argmax(predictions, axis=1)  # Calculate here
    
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices[:num_examples]):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}\nPred: {predicted_classes[idx]}")
        plt.axis('off')
    plt.savefig('misclassified_examples.png')
    plt.close()

def main():
    # Load model and data
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
    current_dir,  # model_training/
    "..",         # go up to neural_network/
    "backend", 
    "model", 
    "digit_classifier.keras") 

    model = tf.keras.models.load_model(model_path)
    x_test, y_test = load_data()
    
    # Run evaluation
    results = evaluate_model(model, x_test, y_test)
    
    # Print results
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Inference Time for 10,000 images: {results['inference_time']:.2f}s")
    print("\nClassification Report:")
    print(results['report'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Save misclassified examples
    if len(results['misclassified']) > 0:
        predictions = model.predict(x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Pass the raw predictions array, not the argmax results
        save_misclassified_examples(x_test, y_test, predictions, results['misclassified'])

if __name__ == '__main__':
    main()