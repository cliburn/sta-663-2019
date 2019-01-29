# STA 663 (2019)

## Learning objectives

- Develop fluency in Python for scientific computing
- Explain how common statistical algorithms work
- Construct models using probabilistic programming
- Implement, test, optimize, and package a statistical algorithm

## Grading

- Homework 40%
- Midterm 1 15%
- Midterm 2 15%
- Project 30%

## Point range for letter grade

- A 94 - 100
- B 85 - 93
- C 70 - 85
- D Below 70

----

## Develop fluency in Python for scientific computing

### Jupyter and Python

- Introduction to Jupyter
- Using Markdown
- Magic functions
- REPL
- Data types
- Operators
- Collections
- Functions and methods
- Control flow
- Packages and namespace
- Coding style
- Understanding error messages
- Getting help
- Saving and exporting Jupyter notebooks

### Text (`string`, `re`)

- The `string` package
- String methods
- Regular expressions
- Loading and saving text files
- Context managers
- Dealing with encoding errors

### Numerics

- Issues with floating point numbers
- The `math` package
- Constructing `numpy` arrays
- Indexing
- Splitting and merging arrays
- Universal functions - transforms and reductions
- Broadcasting rules
- Sparse matrices with `scipy.sparse`

### Data manipulation

- Series and DataFrames
- Creating, loading and saving DataFrames
- Basic information
- Indexing
- Method chaining
- Selecting rows and columns
- Transformations
- Aggregate functions
- Split-apply-combine
- Window functions
- Hierarchical indexing
- Piping with `dfply`

### Graphics

- Graphics from the group up with `matplotlib`
- Statistical visualizations with `seaborn`
- Grammar of graphics with `altair`
- Building dashboards with `dash`

### Functional programming in Python (`operator`, `functional`, `itertoools`, `toolz`)

- Writing a custom function
- Pure functions
- Anonymous functions
- Lazy evaluation
- Higher-order functions
- Decorators
- Partial application
- Using `operator`
- Using `functional`
- Using `itertools`
- Pipelines with `toolz`

### Midterm 1 (15%) 01 Feb

## Explain how common statistical algorithms work

### Data structures, algorithms and complexity

- Sequence and mapping containers
- Using `collections`
- Sorting
- Priority queues
- Working with recursive algorithms
- Tabling and dynamic programing
- Time and space complexity
- Measuring time
- Measuring space

### Solving linear equations

- Solving $Ax = b$
- Gaussian elimination and LR decomposition
- Symmetric matrices and Cholesky decomposition
- Geometry of the normal equations
- Gradient descent to solve linear equations
- Using `scipy.linalg`

### Singular Value Decomposition

- Change of basis
- Spectral decomposition
- Geometry of spectral decomposition
- The four fundamental subspaces of linear algebra
- The SVD
- Geometry of spectral decomposition
- SVD and low rank approximation
- Using `scipy.linalg`

### Optimization I

- Root finding
- Univariate optimization
- Geometry and calculus of optimization
- Gradient descent
- Batch, mini-batch and stochastic variants
- Improving gradient descent
- Root finding and univariate optimization with `scipy.optim`

### Optimization II

- Nelder-Mead (Zeroth order method)
- Line search methods
- Trust region methods
- IRLS
- Lagrange multipliers, KKT and constrained optimization
- Multivariate optimization with `scipy.optim`

### Dimension reduction

- Matrix factorization - PCA and SVD, MMF
- Optimization methods - MDS and t-SNE
- Using `sklearn.decomposition` and `sklearn.manifold`

### Interpolation

- Polynomial
- Spline
- Gaussian process
- Using `scipy.interpolate`

### Clustering  

- Partitioning (k-means)
- Hierarchical (agglomerative Hierarchical Clustering)
- Density based (dbscan, mean-shift)
- Model based (GMM)
- Self-organizing maps
- Cluster initialization
- Cluster evaluation
- Cluster alignment (Munkres)
- Using `skearn.cluster`

### Midterm 2 (15%) 01 March 2019

## Construct models using probabilistic programming

### Probability and random processes

- Working with probability distributions
- Using `random`
- Using `np.random`
- Using `scipy.statistics`
- Simulations

### Monte Carlo methods

- Sampling from data
- Bootstrap
- Permutation resampling
- Sampling from distributions
- Rejection sampling
- Importance sampling
- Monte Carlo integration
- Density estimation

### MCMC

- Bayes theorem and integration
- Numerical integration (quadrature)
- MCMC concepts
- Makrov chains
- Metropolis-Hastings random walk
- Gibbs sampler

### Hamiltonian Monte Carlo

- Hamiltonian systems
- Integration of Hamiltonian system dynamics
- Energy and probability distributions
- HMC
- NUTS

### Probabilistic programming

- Domain-specific languages
- Multi-level Bayesian models
- Using `daft` to draw plate diagrams
- Using `pymc`
- Using `pystan`

### Using `tesnorflow.probability`

- TensorFlow basics
- Distributions and transformations
- Building probabilistic models with `Edward2`

## Implement, test, optimize, and package a statistical algorithm

### Testing

- Why test?
- Test-driven development
- Using `doctest` as documentation
- Using `pytest` to run unit tests
- Using `hypothesis` to auto-generate test cases
- Functional and integration testing
- Always add test if error found

### Packaging and distribution

- Python modules
- Organization of a module
- Writing the setup script
- The Python Package Index
- Package managers
- Containers

### Code optimization I

- Data structures and algorithms
- Vectorization
- JIT compilation with `numba`
- AOT compilation with `cython`

### Code optimization II

- Interpreters and compilers
- Review of C++
- Wrapping C++ functions with `pybind11`

### Parallel programming

- Parallel, concurrent, asynchronous, distributed
- Threads and processes
- Shared memory programming pitfalls: deadlock and race conditions
- Embarrassingly parallel programs with `concurrent.futures` and `multiprocessing`
- Map-reduce
- Master-worker
- Using `ipyparallel` for interactive parallelization

### Final Project (30%)