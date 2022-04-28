# ML-HW2-Chance-of-Adimit

## Brief

11020EE655000 Machine Learning homework 2 using Maximum Likelihood (ML) and Bayesian linear regression methods to train a linear model in order to predict the chance of being admitted to graduate admissions

## Problem definition

Feature vector given as:
$$
\phi(x) = [\phi_1(x),\phi_2(x),......\phi_P(x),\phi_{P+1}(x),\phi_{P+2}(x)]^T
$$
Where:  

- $\phi_{P+1}(x) = x_3 (Research Experience)$
- $\phi_{P+2}(x) = 1   (Bias)$

Gaussian basis function defined as follow:
$$
\phi_k(x)=\exp(-\frac{(x_1-\mu_i)^2}{2s_1^2}-\frac{(x_2-\mu_j)^2}{2s_2^2})
$$
for $$1\leqq i \leqq O_1, 1\leqq j \leqq O_2$$

## Generate $\Phi_k$

In oder to build our weight matrices, we will need to constuct a *design matrix*. With the equations given from the textbook, we are able to build $\Phi$

## Maximum likelihood regression

As shown in the reference video, we see that Ordinary Linear Regression(OLR) derives the same result of Maximum likelihood regression when taking the logarithm of the two. As shown from the testbook, we see that by taking the log likelihood and setting the gradient to 0, we will solve for $w_{ML}$, which is the weight matrix cooresponding to our individual features. By obtaining the weight, we complete the MLR and can do predictions with the resulting weights.

## Baysian linear regression

It seems that these two method share some commonalities, if we look at the log likelihood of the two, we see that they only differ from the hyper-parameters $\alpha$ and $\beta$. If we conduct *Regularized Least Squares*, the simalarities between it and the *Baysian* method is even more obvious. Thus, by carefully choosing the values of $\alpha$ and $\beta$ we can then get somewhat similar results with the two?

## References

1. <https://medium.com/jackys-blog/bayesian-linear-regression-in-python-%E8%B2%9D%E8%91%89%E6%96%AF%E7%B7%9A%E6%80%A7%E8%BF%B4%E6%AD%B8-%E4%B8%8A%E9%9B%86-a0be91a55ffe>
2. <https://medium.com/jackys-blog/bayesian-linear-regression-in-python-%E8%B2%9D%E8%91%89%E6%96%AF%E7%B7%9A%E6%80%A7%E8%BF%B4%E6%AD%B8-%E4%B8%8B%E9%9B%86-7ca47b920bfe>
3. <https://medium.com/@madoria/how-to-ridge-and-lasso-a-mini-python-tutorial-36f6487033e9>
4. <https://medium.com/@lucas.moncada08/linear-regression-in-python-f94f9b72c282>
5. <https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-1-7d0ad817fca5>
6. <https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e>
7. <https://www.youtube.com/watch?v=p6z22Lx6BMU&list=PLdxWrq0zBgPVjvYGoxlc2A5vIpO9NQvw3&index=4&ab_channel=ProfessorKnudson>
