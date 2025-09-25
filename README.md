# Skiptrace Analysis

Analysis performed for a top player in Real State Data Sourcing company

---

## Overview

This project analyzes skiptrace provider sequences to determine the most cost-effective ordering strategy.  
The primary goal was to maximize unique lead retrieval while minimizing associated costs.  

The dataset used in this analysis is private and cannot be shared, as it was sourced from the company’s internal operations.

---

## Problem Statement

- Data: A list of leads and contact information from six providers (T1–T6).  
- Objective: Identify the optimal sequence of providers to maximize unique contacts while controlling costs.  
- Challenge: Providers differ in per-hit costs, match rates, and data depth.  
- Framing: This is an optimization problem, balancing two competing objectives:  
  - Resource constraint: total monetary cost  
  - Target outcome: number of unique leads obtained  

Formally, the goal can be expressed as minimizing the cost per unique contact:

$$
\text{Minimize: } \quad \frac{C}{U}
$$

where  
- \( C \) = cumulative cost of providers used  
- \( U \) = total number of unique contacts retrieved  

---

## Data Preparation

1. Standardization of columns such as `PropertyID`, `PropertyAddress`, `PropertyZip`.  
2. Cleaning identifiers, ensuring numeric consistency and removing duplicates.  
3. Verification of duplicate `PropertyID` across providers (none were found).  
4. Matrix construction:  
   - Rows represent properties  
   - Columns represent providers (T1–T6)  
   - Entry = 1 if provider has contact data, 0 otherwise  

Example:

| PropertyID | T1 | T2 | T3 | T4 | T5 | T6 |
|------------|----|----|----|----|----|----|
| 1511993    | 1  | 1  | 1  | 1  | 1  | 1  |
| 1562732    | 1  | 1  | 0  | 1  | 1  | 1  |

5. Cost dictionary: a mapping from provider to per-hit cost, used as the financial input to optimization.

---

## Methods

### 1. Brute Force
- Explores all permutations of provider sequences (6! = 720).  
- Guarantees the global optimum but scales factorially with the number of providers.  

Process:
1. Generate all possible sequences.  
2. For each sequence, calculate cumulative unique contacts and costs step by step.  
3. Compute final cost-per-contact.  
4. Select sequence with minimum value.  

---

### 2. Greedy Heuristic
- Iteratively selects the provider with best marginal efficiency.  

Efficiency metric:

$$
\text{Efficiency}(T_i) = \frac{\Delta U_i}{c_i}
$$

where  
- $\( \Delta U_i \)$ = new unique contacts gained if provider $\( T_i \)$ is chosen next  
- $\( c_i \)$ = cost of provider $\( T_i \)$  

Process:
1. Start with empty sequence.  
2. At each step, compute efficiency for all remaining providers.  
3. Choose provider with highest efficiency.  
4. Repeat until no new contacts are found.  

---

### 3. Genetic Algorithm
- Inspired by natural selection and genetic recombination.  
- Produces near-optimal solutions with lower computational cost than brute force.  

Process:
1. Start with a random population of sequences.  
2. Evaluate fitness:  

$$
f(S) = \frac{1}{\text{Cost per Contact}(S)}
$$

3. Select best sequences as parents.  
4. Generate new sequences through crossover and mutation.  
5. Iterate for many generations until convergence.  

---

## Analysis

The problem is a form of combinatorial optimization, closely related to the Set Cover Problem.  

Mathematical representation:

$$
\min_{\pi \in \Pi} \; \frac{\sum_{i=1}^{n} c_{\pi(i)}}{\left|\bigcup_{i=1}^{n} U_{\pi(i)}\right|}
$$

where  
- $\( \pi \)$ = sequence of providers  
- $\( c_{\pi(i)} \)$ = cost of provider at position \( i \)  
- $\( U_{\pi(i)} \)$ = set of unique contacts retrieved  

---

## Conclusion

- Brute force guarantees the optimal solution but is computationally expensive.  
- Greedy and genetic methods are more efficient and approximate the optimal solution effectively.  
- The approach demonstrates how mathematical optimization and heuristics can solve real-world cost–coverage trade-offs in lead sourcing.  
