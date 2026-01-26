#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)
#let definition = thmplain("definition", "Definition")
#let theorem = thmplain("theorem", "Theorem")
#let algorithm = thmplain("algorithm", "Algorithm")
#let lemma = thmplain("lemma", "Lemma")
#let proof = thmproof("proof", "Proof")

#definition[Euler--Mascheroni Constant][
  $
    C & := lim_(n -> infinity) (sum_(k=1)^n 1/k - log n) \
      & approx 0.57721566490153286060651209008240243104215933593992
  $
]
#definition[
  $
    z^k C[[z]] & := {z^k sum_(n = 0)^infinity a_n z^n | forall n in NN. a_n in CC} \
    e^k C[[e]] & := {f(e^(i z)) | f(z) in z^k CC[[z]]}
  $
]
#lemma[
  $
  Y_0 (z) & = 2/pi (log(z/2) + C) J_0 (z) + z^2 CC[[z]]
  $
  $forall n in NN.$
$
  Y_n (z)
  = &- ( (1/2 z)^(-n) ) / pi sum_(k = 0)^(n - 1) ((n - k - 1)!)/k! ( (1/4) z^2 )^k \
  &+ (2 / pi) log(z/2) J_n (z) \
  &- ( (1/2 z)^n ) / pi 
  sum_(k = 0)^oo (psi(k + 1) + psi(n + k + 1))
  ((-(1/4) z^2)^k) / (k! (n + k)!)
$
]
#lemma[
  $forall f in e C[[e]].$
  $
  Y_0 (f(z)) 
               & = (J_0 (f(z)))/pi log(4 sin^2(z/2))  + underbrace(2/pi (log abs((f'(0))/2) + C) J_0 (f(z)), = Y_0^((2,f)) (0)) + e^2 C[[e]] \
               &= Y_0^((1,f)) (z) log(4 sin^2(z/2)) + Y_0^((2,f)) (z)
               $
  $forall n in NN.$
  $
  f(z)^n Y_n (f(z)) &= f(z)^n (J_n (f(z)))/pi log(4 sin^2(z/2)) + underbrace(- ( 2^n ) / pi (n - 1)!, = Y_n^((2,f)) (0)) + e^2 C[[e]] \
                &= Y_n^((1,f)) (z) log(4 sin^2(z/2)) + Y_n^((2,f)) (z)
  $
]

For integration of singular functions, one would need to split the function into analytic and singular parts. Both of the following is needed:
- The coefficient (analytic function) of the singular part
- The limit of the remainder (analytic function) evaluated at the singularity

#theorem[Integral of $Y_n$][
  Let $n in NN_0, f in e C[[e]] without e^2 C[[e]], g in e^n C [[e]]$.
  $
    integral_0^(2 pi) g(t) Y_n (f(t)) dd(t) & approx sum_(j = 0)^(N' - 1) g(t_j) (R_j Y_n^((1,f)) (t_j) + w_j Y_n^((2,f)) (t_j)) \
  $
]
#algorithm[
  Assume we have implementation of $J_0, Y_0, f, f'$.
  + Compute ${Y_0 (f(t_j))}_(j = 0)^(N' - 1)$ directly ($Y_0 (f(t_0))$ will be `NaN`).
  + Compute ${Y_n^((1,f)) (t_j)}_(j = 0)^(N' - 1)$.
  + Compute ${Y_n^((2,f)) (t_j)}_(j = 0)^(N' - 1)$ from step 1, 2  ($Y_n^((2,f)) (t_0 = 0)$ will be temporarily `NaN`).
  + Replace $Y_n^((2,f)) (t_0)$.
  + Compute the sum.
]