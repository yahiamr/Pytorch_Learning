
# LeNet-5 with PyTorch

## Overview
This project implements the classic **LeNet-5 convolutional neural network architecture** using PyTorch, focused on **handwritten digit recognition** and trained on the MNIST dataset.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training](#training)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Requirements
* Python 3.x
* PyTorch
* torchvision

## Installation
Install PyTorch and torchvision using pip:
```bash
pip install torch torchvision
```

## Dataset
The **MNIST dataset** - 60,000 training and 10,000 testing images of handwritten digits.

## Model Architecture
LeNet-5 Architecture:
1. **Two Convolutional Layers**
2. **Two Max-Pooling Layers**
3. **Three Fully Connected Layers**

## Usage
Run the script to train the LeNet-5 model on the MNIST dataset.

## Training
- **Epochs:** 10 (adjustable)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

## Code Structure
- `Data Loading:` MNIST dataset loading and transformation.
- `Model Definition:` LeNet-5 architecture.
- `Training Loop:` Training process and parameters update.

## Results
Achieves high accuracy on the MNIST dataset. Extend with validation and confusion matrix for detailed metrics.

## Contributing
Contributions are welcome. Please open an issue or submit a pull request.

## License
Open-source under [MIT License](LICENSE).

## Contact
Open an issue for questions or feedback.

## Acknowledgements
Thanks to Yann LeCun for the original LeNet-5 architecture.
```