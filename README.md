# Drought Prediction using Neural Network

This project aims to gain experience in tuning a neural network for multiclass classification, specifically for predicting drought levels using meteorological data.

## Project Structure

- `droughtPrediction_EDA.ipynb`: Basic Exploratory Data Analysis (EDA).
- `droughtPrediction_DataEng.ipynb`: Combines time series and soil data, removes outliers, splits data into training and testing sets, standardizes features, upsamples using SMOTE, and performs PCA.
- `droughtPrediction_PyTorch_HP.ipynb`: Builds the PyTorch framework, searches for hyperparameters, trains, and saves the final model.
- `droughtPrediction_PyTorch_Final.ipynb`: Shows and evaluates the final model and demonstrates a data pipeline for predicting on the original data format.

## Model Architecture

The final neural network model, `DroughtClassifier`, is structured as follows:

```python
DroughtClassifier(
  (layers): ModuleList(
    (0): Linear(in_features=52, out_features=1024, bias=True)
    (1): Linear(in_features=1024, out_features=512, bias=True)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): Linear(in_features=128, out_features=6, bias=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
)
```

Activation Function: ReLU

### Final Hyperparameters

- Scheduler: `StepLR`
  - `step_size`: 10
  - `gamma`: 0.5
- Dropout Probability: 0.2
- Hidden Layer Sizes: (1024, 512, 256, 128)
- Learning Rate: 0.001

## Model Performance

Metrics on the test dataset:

- Test Loss: 0.6352
- Accuracy: 0.7337
- Macro F1 Mean: 0.6895
- MAE Mean: 0.3255

## Dataset

The data used for this project comes from Kaggle: [US Drought Meteorological Data](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/data).

### About the Dataset

The US drought monitor measures drought across the US, created manually by experts using a wide range of data. The dataset aims to investigate if droughts can be predicted using only meteorological data, potentially allowing generalization of US predictions to other areas of the world.

### Content

The dataset is a classification dataset over six levels of drought, which are:
- No drought
- Five drought levels

Each entry is a drought level at a specific point in time in a specific US county, accompanied by the last 90 days of 18 meteorological indicators.

### Data Splits

To avoid data leakage, the data has been split as follows:

| Split      | Year Range (inclusive) | Percentage (approximate) |
|------------|------------------------|---------------------------|
| Train      | 2000-2009              | 47%                       |
| Validation | 2010-2011              | 10%                       |
| Test       | 2012-2020              | 43%                       |

### Dataset Imbalance

The dataset is imbalanced, which can be observed in the provided graph.  
![image](https://github.com/SantiagoEnriqueGA/droughtPrediction/assets/50879742/f531bd91-8617-43dc-89ce-ab985c0bdea7)

## Results

This project demonstrates the process of tuning a neural network for multiclass classification on a real-world dataset. The steps include data preprocessing, model building, hyperparameter tuning, and evaluation. The final model achieves great results on the test dataset:

| Metric        | Value   |
|---------------|---------|
| Test Loss     | 0.6352  |
| Accuracy      | 0.7337  |
| Macro F1 Mean | 0.6895  |
| MAE Mean      | 0.3255  |

## Final Model Visualization
Simple             |  Detailed
:-------------------------:|:-------------------------:
![image](https://raw.githubusercontent.com/SantiagoEnriqueGA/drought_prediction_pytorch/main/vis/drought_model_simp.png)  |  ![image](https://raw.githubusercontent.com/SantiagoEnriqueGA/drought_prediction_pytorch/main/vis/drought_model_comp.png)

## Model Componenets Explained
1. **Input Layer:**
   - The input tensor has a shape of `(1, 52)`, indicating a batch size of 1 and 52 input features.

2. **First Linear Layer (layers.0):**
   - **weights:** `layers.0.weight (1024, 52)`
   - **bias:** `layers.0.bias (1024)`
   - This layer maps the 52 input features to 1024 features using a linear transformation.

3. **First Activation and Dropout:**
   - The output from the first linear layer passes through a ReLU activation function, followed by a dropout layer to introduce regularization.
   - Represented by `ReLUBackward0` and `TBackward0`.

4. **Second Linear Layer (layers.1):**
   - **weights:** `layers.1.weight (512, 1024)`
   - **bias:** `layers.1.bias (512)`
   - This layer takes the 1024 features from the previous layer and maps them to 512 features.

5. **Second Activation and Dropout:**
   - Similar to the first layer, the output from the second linear layer passes through ReLU activation and dropout.
   - Represented by `ReLUBackward0` and `TBackward0`.

6. **Third Linear Layer (layers.2):**
   - **weights:** `layers.2.weight (256, 512)`
   - **bias:** `layers.2.bias (256)`
   - This layer reduces the 512 features to 256 features.

7. **Third Activation and Dropout:**
   - Again, the output goes through ReLU activation and dropout.
   - Represented by `ReLUBackward0` and `TBackward0`.

8. **Fourth Linear Layer (layers.3):**
   - **weights:** `layers.3.weight (128, 256)`
   - **bias:** `layers.3.bias (128)`
   - This layer further reduces the features from 256 to 128.

9. **Fourth Activation and Dropout:**
   - The output undergoes ReLU activation and dropout.
   - Represented by `ReLUBackward0` and `TBackward0`.

10. **Fifth (Output) Linear Layer (layers.4):**
    - **weights:** `layers.4.weight (6, 128)`
    - **bias:** `layers.4.bias (6)`
    - This final layer maps the 128 features to 6 output classes.

11. **Output:**
    - The final output tensor has a shape of `(1, 6)`, representing the model's prediction probabilities for each of the 6 classes.

**AccumulateGrad Nodes:**
- These nodes represent where the gradients are accumulated during the backward pass for updating the model parameters.

**AddmmBackward0 Nodes:**
- These nodes represent the matrix multiplication operations performed during the forward pass through each linear layer.
- Detailed information about the matrix multiplication is provided, including the sizes of the matrices involved (`mat1`, `mat2`), strides, and saved tensors.

**ReluBackward0 and TBackward0 Nodes:**
- These nodes represent the ReLU activation and dropout operations applied to the output of each linear layer.

**mat1, mat2, and result Tensors:**
- `mat1` and `mat2` represent the input and weight matrices involved in the matrix multiplication.
- `result` represents the output tensor after each matrix multiplication.




