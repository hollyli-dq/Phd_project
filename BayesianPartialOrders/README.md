# Bayesian Inference for Partial Orders

The package is for programming teh bayesian inference for partial orders, we first include some notations and also include the model below.

We will further include the dataset in the data processing step for further application. And also a MCMC has been included. 

The partial order is important as it has recently been viewed that center the random lists on a partial order. Lists are random linear extensions of a partial order or linear extensions observed with noise.

### Symbols:

1. M = [M ], [M ] = {1, . . . , M } be the universe of objects over which preferences can
   be given and suppose there are M = |M|
2. A poset h is a choice set equipped with a partial order ≻h .
3. A linear extension of h is any complete order ℓ ∈ CS satisfying j1 ≻h j2 ⇒ j1 ≻ℓ j2 , so the linear extension “completes” the partial order ≻h .
4. Denote by L[h] the set of all linear extensions of h. For j ∈ S let  Lj [h] = {ℓ ∈ L[h] : max(ℓ) = j}, j is the first element in the list.

### Challenges:

1. Computing |L[h]| is #P-complete (Brightwell and Winkler, 1991), so we cannot evaluate
   pS (y|h) for general h ∈ HS and large m.

## The problem

1. Context-independent preference with
2.

## Methdology

* The likelihood p(Y|h) for a poset h, when we have N lists Y1, ...,Y_N observed on choice sets S_1, ... , S_N. we now give a prior and posterior for h.
* Define a Bayesian model with a specific hierarchical structure involving latent variables and their transformations.
* For α and Σρ defined above, if we take Uj,: ∼ N (0, Σρ ), independent for each j ∈ M,**U**j is sampled from a multivariate normal distribution with mean 0 and covariance matrix Σρ.
* ηj,: = G−1 (Φ(Uj,: )) + αj 1TK ,

  * **G**−**1** represents the inverse of a CDF (likely a Gumbel distribution, given the mention of GG**G**), and Φ\\Phi**Φ** is the CDF of the standard normal distribution.
  * This construction essentially "maps" the latent normal random variable through a series of transformations.
  * ηj,:\\eta\_{j,:}**η**j**,**:**** adds a constant shift by αj\\alpha\_j**α**j****, applied to each component of the vector, meaning that ηj,:\\eta\_{j,:}**η**j**,**:**** is transformed through a non-linear mapping and a shift.
* y ∼ p(·|h(η(U, β))),
  then h(η:,k ) ∼ P L(α; M) for each k = 1, . . . , K. In particular, if K = 1 then y ∼ P L(α; M).
*
