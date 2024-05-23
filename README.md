# KTorch

## Motivation

KTorch is a dedicated repository for machine learning and deep learning algorithms. It contains handcrafted implementations of popular machine learning algorithms, neural networks' layers, an autograd engine, and much more.

## Build Status

- The project is currently in development
- Implemented and Ready-to-use Machine Learning Algorithms, Neural Network layers, and others can be found in the `Features & Algorithms` section
- Under Testing:
  - Transformers, Conv2D, Layer normalization, Batch normalization
  - Ensemble Models, PCA
- Under Development:
  - RNNs, LSTMs, and GRUs

## Tech Stack

<div align="center>
  
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
  
</div>

## Features & Algorithms

The following algorithms are implemented, tested, and used on different datasets, which can be found in [Example Notebooks](./examples_notebooks/):

<details>
<summary>Machine learning algorithms</summary>

- [k-Nearest Neighbors Regressor](./algorithms/neighbors/KNeighborsRegressor.py)
- [k-Nearest Neighbors Classifier](./algorithms/neighbors/KNeighborsClassifier.py)
- [Linear Regression](./algorithms/linear_model/LinearRegression.py)
- [Logistic Regression](./algorithms/linear_model/LogisticRegression.py)
- [Ridge Regression](./algorithms/linear_model/RidgeRegression.py)
- [SGD Regression (Linear/Ridge/Lasso/Elasticnet)](./algorithms/linear_model/SGDRegressor.py)
- [Linear Discriminant Analysis](./algorithms/discriminant_analysis/LinearDiscriminantAnalysis.py)
- [Quadratic Discriminant Analysis](./algorithms/discriminant_analysis/QuadraticDiscriminantAnalysis.py)
- [Gaussian Naive Bayes](./algorithms/naive_bayes/GaussianNB.py)
- [Multinomial Naive Bayes](./algorithms/naive_bayes/MultinomialNB.py)
- [Support Vector Classifier](./algorithms/svm/SVC.py)
- [Decision Tree Regressor](./algorithms/tree/DecisionTreeRegressor.py)
- [Decision Tree Classifier](./algorithms/tree/DecisionTreeClassifier.py)\_
</details>

<details>
  <summary>Autograd Engine</summary>

- [Tensor](./autograd/engine.py) which supports the forward and backward propagations of the following operations:
  - Addition (`__add__`, `__radd__`)
  - Subtraction (`__sub__`, `__rsub__`)
  - Multiplication (`__mul__`, `__rmul__`)
  - Division (`__truediv__`)
  - Power (`__pow__`, `square`)
  - Negation (`__neg__`)
  - Matrix multiplication (`__matmul__`)
  - Relu (`ReLU`)
  - Sigmoid (`sigmoid`)
  - Tanh (`tanh`)
  - Exponential (`exp`)
  - Logarithm (`log`)
  - Sum (`sum`)
  - Mean (`mean`)
  - Variance (`var`)
  - Maximum (`max`)
  - Minimum (`min`)
  - Cosine (`cos`)
  - Sine (`sin`)
  - Greater Than (`__gt__`)
  - Cumulative Distribution Function (`phi`)
  - Split (`split`)
  - Masked Fill (`masked_fill`)
  - Softmax (`softmax`)
  - Unsqueeze (`unsqueeze`)
  - Squeeze (`squeeze`)
  - Transpose (`transpose`)
  - 2D Convolution (`conv2d`)
  - Padding (`pad`)
  - Flip (`flip`)
  - Concatenate (`cat`)
  - Clamp (`clamp`)
  - One Hot Encoding (`one_hot`)
  - Absolute (`abs`)
  - Flatten (`flatten`)
  - View (`view`)
  - Get Item (`__getitem__`)

</details>

<details>
<summary>Neural network layers</summary>

- [Linear Layer](./nn/Linear.py)
- [Sequential](./nn/Sequential.py)
- [Flatten](./nn/Flatten.py)
- [Dropout](./nn/Dropout.py)
- Activation Functions
  - [Tanh](./nn/Tanh.py)
  - [Sigmoid](./nn/Sigmoid.py)
  - [Softmax](./nn/Softmax.py)
  - [ReLU](./nn/ReLU.py)
  - [GELU](./nn/GELU.py)
- Loss Functions
  - [Categorical Crossentropy](./nn/CrossEntropyLoss.py)
  - [Binary Crossentropy](./nn/BCELoss.py)
  - [Binary Crossentropy](./nn/BCEWithLogitsLoss.py) - a numerically stable version that uses log-sum-exp trick
  - [Mean Squared Error](./nn/MSELoss.py)

</details>

<details>
<summary>Optimizers</summary>

- [Stochastic Gradient Descent / SGD with Momentum / SGD with Nestrov Momentum](./optim/SGD.py)
- [RMSProp / Centered RMSProp](./optim/RMSProp.py)
- [AdaDelta](./optim/Adadelta.py)
- [Adagrad](./optim/Adagrad.py)
- [Adam / AMSGrad](./optim/Adam.py)

</details>

<details>
<summary>Tokenizers</summary>

- [Byte Pair Encoding](./tokenizer/BPE.py)

</details>

<details>
<summary>Utilites (Metrics and other stuff)</summary>

</details>

## Credits

This is a list of some of the sources that helped me in learning and making this repo

1. [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
2. [Serrano.Academy](https://www.youtube.com/@SerranoAcademy)
3. [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)
4. [Introduction to Statistical Learning](https://www.statlearning.com/)
5. The Elements of Statistical Learning
6. [Pascal Poupart](https://www.youtube.com/results?search_query=pascal+poupart)

## License

The rest of the software is open source and licensed under the [MIT License](https://opensource.org/license/mit/)
