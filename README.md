# Multinomial-Logistic-Regression
Writing a program that implements a text analyzer using Multinomial Logistic Regression

The file learns the parameters of a multinomial logistic regression model that predicts a tag (i.e. label) for each word and its corresponding feature vector. The program outputs the labels of the training and test examples and calculate training and test error (percentage of incorrectly labeled words).

All model parameters are initialized to 0

SGD is used to optimize the parameters of the model. The number of times SGD loops through all of the training data (num epoch) will be
specified as a command line flag. The learning rate is set at a constant to 0.5

2 different feature structures are used here:
Model 1: The model defines a probability distribution over the current tag using parameter vector and a feature vector based only on the current word. The model should be used when the <feature flag> is set to 1
  
Model 2: The model defines a probability distribution over the current tag using parameter vector and a feature vector based only on the previous word, current word and the next word. The model should be used when the <feature flag> is set to 2
  
The different inputs/outputs to the program are (using sys.arv[n]):
1. train input
2. validation input
3. test input
4. train output
5. test output
6. test and train error as output
7. number of epochs
8. feature flag (1 or 2)

the function 'nll' is used to calculate the negative log likelihood which is to be optimized
the function 'errcal' is to calculate the error for train, test and validation outputs
the function 'model2' tries to give an output array suitable for model 2
