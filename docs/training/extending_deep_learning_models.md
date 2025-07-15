# **Developer Guide: Extending the Deep Learning Framework**

This guide explains how to extend the project's deep learning framework by subclassing the provided base model. The design allows you to integrate custom **PyTorch model architectures** while maintaining consistent training, prediction, and evaluation workflows.  
All training-related scripts, logs, and data outputs are organized under the training/ directory in the project root. Model checkpoints (saved versions of a model’s weights and configuration) are organized under the ai-microservice/ directory in the project root.

## **Table of Contents**

* [What You Can Extend](https://www.google.com/search?q=%23what-you-can-extend)  
* [Responsibilities of Your Subclass](https://www.google.com/search?q=%23responsibilities-of-your-subclass)  
* [Inputs and Outputs](https://www.google.com/search?q=%23inputs-and-outputs)  
  * [Expected Inputs to forward()](https://www.google.com/search?q=%23expected-inputs-to-forward)  
  * [Expected Output](https://www.google.com/search?q=%23expected-output)  
* [What the Base Class Handles for You](https://www.google.com/search?q=%23what-the-base-class-handles-for-you)  
* [Subclassing Assumptions](https://www.google.com/search?q=%23subclassing-assumptions)  
* [Example Subclasses](https://www.google.com/search?q=%23example-subclasses)  
* [Configuration Notes](https://www.google.com/search?q=%23configuration-notes)  
* [Further Reading](https://www.google.com/search?q=%23further-reading)  
* [Summary: What You Must Do](https://www.google.com/search?q=%23summary-what-you-must-do)

## **What You Can Extend**

To integrate your own custom deep learning model into the framework, you need to provide **two key classes**:

1. **A Wrapper Class:** This class must subclass base\_deep\_learner.BaseDeepLearningClassifier. Its primary role is to implement the \_build\_model\_architecture() method, which will return an instance of your custom PyTorch neural network.  
2. **A Custom Neural Network Class:** This class defines the actual architecture of your deep learning model. It must subclass torch.nn.Module and implement the model's layers and its forward() pass logic.

## **Responsibilities of Your Subclass**

When creating your wrapper class, you **must implement** the following abstract method:

### **\_build\_model\_architecture(self) \-\> torch.nn.Module**

This method is invoked by the base class during the training setup phase. It is responsible for instantiating and returning your custom neural network model.  
def \_build\_model\_architecture(self):  
    class MyNetwork(nn.Module):  
        def \_\_init\_\_(self):  
            super().\_\_init\_\_()  
            \# Define your model's layers here (e.g., convolutional, recurrent, dense layers)

        def forward(self, text\_inputs, categorical\_inputs, numerical\_inputs):  
            \# Implement the forward pass logic for your network.  
            \# This method processes the inputs and produces the raw logits.  
            return logits

    return MyNetwork()

## **Inputs and Outputs**

Your custom neural network class's forward() method must adhere to a specific input and output signature to ensure compatibility with the base framework.

### **Expected Inputs to forward()**

Your model's forward() method must accept the following keyword arguments, each being a dictionary of PyTorch tensors:  
def forward(self,  
            text\_inputs: Dict\[str, Tensor\],  
            categorical\_inputs: Dict\[str, Tensor\],  
            numerical\_inputs: Dict\[str, Tensor\]) \-\> Tensor

* text\_inputs: A dictionary where keys are the names of text features (e.g., 'RequestURIPath', 'RequestBody') and values are PyTorch tensors, each with shape (batch\_size, sequence\_length).  
* categorical\_inputs: A dictionary where keys are the names of categorical features (e.g., 'RequestMethod') and values are PyTorch tensors, each with shape (batch\_size, 1).  
* numerical\_inputs: A dictionary where keys are the names of numerical features and values are PyTorch tensors, each with shape (batch\_size, 1).

### **Expected Output**

The forward() method must return a tensor representing the raw logits (pre-activation scores) for classification:

* For **multi-class classification**, the output tensor should have the shape (batch\_size, num\_classes).  
* For **binary classification** (especially if using torch.nn.BCEWithLogitsLoss), the output tensor should have the shape (batch\_size,) or (batch\_size, 1).

## **What the Base Class Handles for You**

The BaseDeepLearningClassifier abstract class provides significant boilerplate, handling common machine learning tasks automatically:

* **Device Management:** Automatically moves tensors and models to the appropriate device (GPU or CPU) based on availability and configuration.  
* **Label Encoding:** Manages the encoding of target labels using LabelEncoder.  
* **Loss and Optimizer Setup:** Configures the loss function and optimizer based on parameters provided in the configuration.  
* **Batching and DataLoader Creation:** Handles the creation of DataLoader instances for efficient batch processing during training and evaluation.  
* **Training Loop and Logging:** Manages the iterative training process, including forward/backward passes, optimizer steps, and logging of training metrics.  
* **Prediction Methods:** Provides standard predict and predict\_proba methods for inference, abstracting away the model's raw output.

## **Subclassing Assumptions**

To ensure seamless integration and proper functioning, adhere to these assumptions when subclassing:

1. You **must not override** the fit, predict, or predict\_proba methods of the base class unless there is an absolute, critical need for custom behavior that cannot be achieved otherwise.  
2. The signature of your custom model's forward method **must exactly match** the interface described in the "Expected Inputs to forward()" section.  
3. If you require a custom loss function or optimizer, specify its type and parameters through the loss\_type, loss\_params, optimizer\_type, and optimizer\_params arguments in your wrapper class's constructor.

## **Example Subclasses**

For practical examples of how to implement wrapper and custom neural network classes, refer to:

* [MultiInputMLPClassifier](https://www.google.com/search?q=./training/MLP_wrapper.py)  
* [MultiInputCNNClassifier](https://www.google.com/search?q=./training/CNN_wrapper.py)

## **Configuration Notes**

The behavior and hyperparameters of your custom model are configured through arguments passed during its instantiation. These parameters are typically defined in the project's config.py file.  
For example, if you choose a mnemonic of **rnn** for your architecture, ensure the following elements are present and correctly configured in **config.py**:  
PERFORM\_DENSE\_PREPROCESSING  \= True  
\# ...  
MODEL\_TYPE \= 'rnn' \# \<---- Your chosen mnemonic, matching a key in MODEL\_CLASSES and MODEL\_PARAMS  
\# ...  
MODEL\_CLASSES \= {  
    \# ... other models ...  
    'rnn': 'RNNClassifier' \# \<---- Your wrapper class name (e.g., the class that implements \_build\_model\_architecture)  
    \# ...  
}  
\# ...  
MODEL\_PARAMS \= {  
    \# ... other model parameters ...  
    'rnn': {  
        \# Learning parameters  
        'learning\_rate': 0.001,  
        'epochs': 50,  
        'batch\_size': 32,  
        'optimizer\_type': 'adam', \# Options: 'adam' or 'sgd'  
        'optimizer\_params': {  
            'weight\_decay': 0.0001,  
        },  
        'loss\_type':  'CrossEntropyLoss', \# Example: 'CrossEntropyLoss', 'BCEWithLogitsLoss'  
        'loss\_params': {  
            \# Parameters specific to your chosen loss function, if any  
        },  
        'dropout\_rate': 0.5,  
        'hidden\_size': 64,  
        'num\_classes': 2, \# Number of output classes for your classification task  
        'numerical\_hidden\_size': 32,  
        'text\_embed\_dims': {  
            'RequestURIPath': 32,  
            'RequestURIQuery': 32,  
            'RequestBody': 32,  
            'UserAgent': 32  
        },  
        'categorical\_embed\_dims': {  
            'RequestMethod': 3  
        },  
        'text\_rnn\_configs': { \# Specific to RNN models  
            'RequestURIPath': {'hidden\_size': 128, 'num\_layers': 1, 'bidirectional': False},  
            'RequestURIQuery': {'hidden\_size': 128, 'num\_layers': 1, 'bidirectional': False},  
            'RequestBody': {'hidden\_size': 128, 'num\_layers': 1, 'bidirectional': False},  
            'UserAgent': {'hidden\_size': 128, 'num\_layers': 1, 'bidirectional': False}  
        },  
        'rnn\_type': 'GRU',  \# Options: 'LSTM' or 'GRU'  
    }  
}  
\# ...  
TUNING\_PARAMS \= {  
    \# ... other tuning parameters ...  
    'rnn': {  
         'hidden\_size': \[32\], \# Example: \[32, 64\] for hyperparameter search  
         'learning\_rate': \[0.01\], \# Example: \[0.001, 0.01\]  
    }  
    \# ...  
}

## **Further Reading**

To deepen your understanding of building custom neural network models in PyTorch, refer to the official PyTorch documentation and tutorials:

* [Learn the Basics: Build Models with nn.Module](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)  
* [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)  
* [PyTorch nn.Module documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

## **Summary: What You Must Do**

| Requirement | Your Responsibility |
| :---- | :---- |
| Subclass BaseDeepLearningClassifier | ✅ Yes |
| Implement \_build\_model\_architecture() | ✅ Yes |
| Return a valid torch.nn.Module instance | ✅ Yes |
| Accept the correct inputs in forward() | ✅ Yes |
| Produce logits for classification | ✅ Yes |
| Leave fit, predict, predict\_proba alone | ✅ Yes |
| Add configuration parameters in config.py | ✅ Yes |
