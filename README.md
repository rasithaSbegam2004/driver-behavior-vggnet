# Driver Behavior Detection using CNN (VGGNet)

This project implements a Convolutional Neural Network (CNN) based on the VGGNet architecture to classify driver behaviors from images into five categories: Other, Safe Driving, Talking on Phone, Texting on Phone, and Turning.


# Dataset
Each folder contains images corresponding to the respective driver behavior.



## Requirements

- Python 3.10+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-image
- OpenCV

Install dependencies with:

```sh
pip install tensorflow keras numpy pandas matplotlib scikit-image opencv-python
```

## Usage

1. Download the dataset and update the paths in the notebook if necessary.
2. Open `vggnet.ipynb` in Jupyter Notebook or VS Code.
3. Run all cells to:
   - Load and preprocess the data
   - Build and train the VGGNet model
   - Visualize training/validation loss and accuracy

## Model

The model is a custom VGGNet-like CNN implemented using TensorFlow/Keras. It is trained for 20 epochs and evaluated on validation data.

## Results

The notebook visualizes both training and validation loss and accuracy per epoch, helping to assess model performance and overfitting.

