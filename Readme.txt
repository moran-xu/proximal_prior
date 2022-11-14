The following code can be used to reproduce the results in 'Bayesian Inference Using the Proximal Mapping: Uncertainty Quantification Under Varying Dimensionality'.

Abstract: In statistical applications, it is common to encounter parameters supported on a
varying or unknown dimensional space. Examples include the fused lasso regression,
the matrix recovery under an unknown low rank, etc. Despite the ease of obtaining
a point estimate via optimization, it is much more challenging to quantify their un-
certainty. In the Bayesian framework, a major difficulty is that if assigning the prior
associated with a p-dimensional measure, then there is zero posterior probability on
any lower-dimensional subset with dimension d < p. To avoid this caveat, one needs
to choose another dimension-selection prior on d, which often involves a highly com-
binatorial problem. To significantly reduce the modeling burden, we propose a new
generative process for the prior: starting from a continuous random variable such as
multivariate Gaussian, we transform it into a varying-dimensional space using the
proximal mapping. This leads to a large class of new Bayesian models that can di-
rectly exploit the popular frequentist regularizations and their algorithms, such as
the nuclear norm penalty and the alternating direction method of multipliers, while
providing a principled and probabilistic uncertainty estimation. We show that this
framework is well justified in the geometric measure theory, and enjoys a convenient
posterior computation via the standard Hamiltonian Monte Carlo. We demonstrate
its use in the analysis of the dynamic flow network data.

To reproduce the results in our manuscript:

Figure 1 is not a simulated result --- it is a concept figure drawn via draw.io. 

Figure 2 can be replicated through lam_prior_Figure2.R

Figure 3 can be replicated through set_expansion_Figure3.py

Figure 4 can be replicated through the following files: 
traffic_Figure4(a).py; 
trafficFigure4(b).py;  
trafficFigure4(c).py;
trafficFigure4(d)(e).py;
traffic_Figure4(f).py; 
traffic_Figure4(g).py; 
The data for Figure 4 are in edges.csv and nodes.csv.