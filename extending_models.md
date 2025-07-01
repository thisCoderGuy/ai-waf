# Developer Guide: Extending the Deep Learning Framework

This guide explains how to extend the codebase by subclassing the provided base model class. The design allows you to plug in custom **PyTorch model architectures** while retaining consistent training, prediction, and evaluation workflows.

All training-related scripts, logs, and data outputs are organized under the `training/` directory in the project root.

Model checkpoints (saved versions of a model’s weights and configuration) are organized under the `ai-microservice/` directory in the project root. 

---

## What You Can Extend

To integrate your own model, you need to provide **two classes**:

1. **A wrapper class** that subclasses:

   ```python
   base_deep_learner.BaseDeepLearningClassifier
   ```

   This class implements the method `_build_model_architecture()` to return your custom PyTorch model.

2. **A custom neural network class** that subclasses:

   ```python
   torch.nn.Module
   ```

   This class implements the actual model layers and the `forward()` method.

## Responsibilities of Your Subclass

### You **must implement**:

#### `_build_model_architecture(self) -> torch.nn.Module`

This abstract method **must return a `torch.nn.Module` instance** that defines your deep learning model architecture. It is invoked by the base class during training setup.

```python
def _build_model_architecture(self):
    class MyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Define layers here

        def forward(self, text_inputs, categorical_inputs, numerical_inputs):
            # Implement forward logic
            return logits

    return MyNetwork()
```

---

## Inputs and Outputs

### Expected Inputs to `forward()`

Your model's `forward()` method must accept the following keyword arguments:

```python
def forward(self,
            text_inputs: Dict[str, Tensor],
            categorical_inputs: Dict[str, Tensor],
            numerical_inputs: Dict[str, Tensor]) -> Tensor
```

Each input is a **dictionary of PyTorch tensors**:

- `text_inputs`: A dictionary mapping each text feature name to a tensor of shape `(batch_size, seq_len)`
- `categorical_inputs`: A dictionary mapping each categorical feature name to a tensor of shape `(batch_size, 1)`
- `numerical_inputs`: A dictionary mapping each numerical feature name to a tensor of shape `(batch_size, 1)`

### Expected Output

The output of `forward()` must be a tensor of shape:

- `(batch_size, num_classes)` for multi-class classification
- `(batch_size,)` or `(batch_size, 1)` for binary classification (if using `BCEWithLogitsLoss`)

---

## What the Base Class Handles for You

- Device management (GPU or CPU)
- Label encoding using `LabelEncoder`
- Loss and optimizer setup from config
- Batching and DataLoader creation
- Training loop and logging
- Prediction methods (`predict`, `predict_proba`)

---

## Subclassing Assumptions

1. You must **not override `fit`**, **`predict`**, or **`predict_proba`** unless absolutely necessary.
2. The `forward` method must match the interface above.
3. If you are using a custom loss or optimizer, specify it through the constructor parameters:
   - `loss_type`, `loss_params`
   - `optimizer_type`, `optimizer_params`

---

## Example Subclasses

[MultiInputMLPClassifier](./training/MLP_wrapper.py)

[MultiInputCNNClassifier](./training/CNN_wrapper.py)

---

## Configuration Notes

You must configure model behavior using arguments passed during instantiation. These  come from `config.py`:

For example, if you choose a mnemonic of **`rnn`** for your architecture, make sure to  have the elements below in **`config.py`**

```python
PERFORM_DENSE_PREPROCESSING  = True
...
MODEL_TYPE = 'rnn' # <---- Your chosen mnemonic
...
MODEL_CLASSES = {
    ...
    'rnn': 'RNNClassifier' # <---- Your wrapper class name
    ...
}
...
MODEL_PARAMS = {
    ...
    'rnn': { 
        # Learning parameters
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'optimizer_type': 'adam', # adam or sgd
        'optimizer_params': {
            'weight_decay': 0.0001, 
        },
        'loss_type':  'CrossEntropyLoss', 
        'loss_params': {
        },
        'dropout_rate': 0.5, 
        'hidden_size': 64,
        'num_classes': 2,
        'numerical_hidden_size': 32,
        'text_embed_dims': {
            'RequestURIPath': 32,
            'RequestURIQuery': 32,
            'RequestBody': 32,
            'UserAgent': 32
        },
        'categorical_embed_dims': {
            'RequestMethod': 3
        },
        'text_rnn_configs': {
            'RequestURIPath': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'RequestURIQuery': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'RequestBody': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'UserAgent': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False}
        },
        'rnn_type': 'GRU',  # LSTM or GRU
    }
}
    ...
TUNING_PARAMS = {
    ...
    'rnn': {
         'hidden_size': [32], # [32, 64],
         'learning_rate': [0.01], #0.001, 0.01],
    }
    ...
}

```

---

## Further Reading

To understand how to build custom neural network models in PyTorch, refer to:

- [Learn the Basics: Build Models with nn.Module](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
- [PyTorch nn.Module documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

---

## Summary: What You Must Do

| Requirement                                   | Your Responsibility  |
| --------------------------------------------- | --------------------- 
| Subclass `BaseDeepLearningClassifier`         | ✅ Yes               |
| Implement `_build_model_architecture()`       | ✅ Yes               |
| Return a valid `nn.Module`                    | ✅ Yes               |
| Accept the correct inputs in `forward()`      | ✅ Yes               |
| Produce logits for classification             | ✅ Yes               |
| Leave `fit`, `predict`, `predict_proba` alone | ✅ Yes               |
| Add configuration parameters in `config.py`   | ✅ Yes               |

