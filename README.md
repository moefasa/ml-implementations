# ml-implementations
Implementations of various machine learning algorithms. 

## Description

This repo contains implementations for a linear support vector machine as well as a parallel one-vs-rest multiclassifier. 

## Examples

The examples folder contains 3 jupyter notebooks with demo's of the estimators in the libaries.

* ml-implementations-demo-1: Demonstrates the use of the MyLinearSVM and MyOneVsRestClassifier classes on simulated data.
* ml-implementations-demo-2: Demonstrates the use of the MyLinearSVM and MyOneVsRestClassifier classes on a real-world dataset.
* Scikit-Learn Comparison: Tests MyLinearSVM against scikit-learn's LinearSVC with respect to accuracy and speed.

## Libraries

* **base_estimators**: Contains the MyLinearSVM class and accompanying functions.
* **gradient_utils**: Implements fast gradient descent for convex optimization problems. See the Contributing section for more info.
* **loss_functions**: Contains various loss functions that can be implemented by the gradient_utils library.
* **multiclass_estimators**: Contains the MyOneVsRestClassifier and tools for distributed computing.

## Contributing

### gradient_utils

To use the **gradient_utils** library you just need to call the GradientOptimizer class. For example,
```python
optimizer = GradientOptimizer(loss_function='squared-hinge')
optimizer.optimize(X, y, lambda) # lambda is an optiona argument.
```
The optimizer will then run fast gradient descent and return an a list of parameter values at each iteration of gradient descent.

 ### loss_functions

To add a loss function you'll need to inherit from the LossFunction class and implement 3 methods: set_space, obj, and compute_grad,
and be sure to save the dimensions of X (n, d) as self.n, self.d. For example,

```python
class MyLossFunction(LossFunction):
    def set_space(self, X, y, lambduh=1):
        self.X = X
        self.y = y
        self.l = lambduh
        self.n, self.d = n, d
        return self

    def obj(self, beta):
        X = self.X
        y = self.y
        l = self.l
        n, d = self.n, self.d

        term = (y.T @ y) - 2*(X @ beta).T) @ y + (X @ beta).T @ X @ beta
        
        return (1.0/n)*term  + l*(norm(beta)**2)
     ...
```
Note that the set_space section should be used to cache any constants used to compute objective or gradient. See an example of this in the SquaredHingeLoss loss function.

You'll also need to add a key to your loss function in the module-level get_available_functions dictionary. This will allow it to be come available from base estimators. For example, 'squared-hinge' calls the SquaredHingeLoss class.
