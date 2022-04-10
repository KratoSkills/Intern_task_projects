# Multivariate Normal (MVN) distribution using JAX

## *Problem Statement: -*
Implemented from scratch a sampling method to draw samples from a multivariate Normal (MVN) Distribution using JAX in Python.
**Restriction -** Use only ***jax.random.uniform*** and not ***jax.random.normal***

## *Theory: -*

* **[Normal Distribution](https://www.investopedia.com/terms/n/normaldistribution.asp):** Normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve.

* **[Multivariate Normal Distribution](https://brilliant.org/wiki/multivariate-normal-distribution/):** A multivariate normal distribution is a vector in multiple normally distributed variables, such that any linear combination of the variables is also normally distributed. It is mostly useful in extending the central limit theorem to multiple variables, but also has applications to bayesian inference and thus machine learning, where the multivariate normal distribution is used to approximate the features of some characteristics; for instance, in detecting faces in pictures.

## *Libraries Used: -*
* **[JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)** - JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.
* **[SciPy](https://scipy.org/)** - SciPy provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics and many other classes of problems.
* **[scikit-learn](https://scikit-learn.org/stable/)** - Simple and efficient tools for predictive data analysis. Accessible to everybody, and reusable in various contexts. Built on NumPy, SciPy, and matplotlib.

## *Steps in Solution: -*
1. Randomly create Mean and Covariance Matrix.
2. Implementing Sampling Function.
3. Check whether generated (Mean, Covariance) and given (Mean, Covariance) are equivalent or not.

## *References: -*
* [Multivariate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
* [Uniform Distribution in jax.random](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.uniform.html)
* [JAX's NUMPY](https://jax.readthedocs.io/en/latest/jax.numpy.html)
* [Spatial Distance in SciPy](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
