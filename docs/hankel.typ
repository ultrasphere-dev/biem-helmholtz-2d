#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)
#let definition = thmplain("definition", "Definition")
#let theorem = thmplain("theorem", "Theorem")
#let lemma = thmplain("lemma", "Lemma")
#let proof = thmproof("proof", "Proof")
#let span = $op("span")$
#let sign = $op("sign")$
#let hk = $H^((1))$

= Integration of $hk_0$

#definition[Euler--Mascheroni Constant][
  $
  C &:= lim_(n -> infinity) (sum_(k=1)^n 1/k - log n) \
  &approx 0.57721566490153286060651209008240243104215933593992
  $
]
#lemma[
  $
  J_0 (z) &= sum_(k = 0)^infinity b_k (z/2)^(2 k), b_k = (-1)^k/(k!)^2 \
  N_0 (z) &= 2/pi (log z/2 + C) J_0 (z) + sum_(k = 1)^infinity b_k (z/2)^(2 k), b_k = (-1)^k/(k!)^2 sum_(m = 1)^k 1/m
  $
]
#definition[
  $
  z^k C[[z]] := {z^k sum_(n = 0)^infinity a_n z^n | forall n in NN. a_n in CC}
  $
]
#lemma[
  $
  J_0 (z) &= 1 + z CC[[z]] \
  N_0 (z) &= 2/pi (log z/2 + C) J_0 (z) + z CC[[z]]
  $
]
#lemma[
  Let $f in z CC[[z]].$
  $
  N_0 (f(z)) &= 1/pi log(4 sin^2 z/2) J_0 (f(z)) + 2/pi (log f(z)/2 - log 2 sin z/2 + C) J_0 (f(z)) + z CC[[z]] \
  &= 1/pi log(4 sin^2 z/2) J_0 (f(z)) + 2/pi (log f(z)/(4sin z/2) + C) J_0 (f(z)) + z CC[[z]] \
  &= N_0^((1)) (z) log (4 sin^2 z/2) + N_0^((2)) (z) \
  $
  where $N_0^((1)), N_0^((2)) in C[[z]]$. Note that analycity of $N_0^((2))$ at $0$ is not obvious but by L'Hospital's rule we have
  $
  lim_(z -> 0) N_0^((2)) (z) = 2/pi (log (f'(z))/2 + C)
  $
]
#theorem[Integral of $hk_0$][
  Let $f in z C[[z]], g in C[[z]]$.
  $
  integral_0^(2 pi) g(t) N_0 (f(t)) dd(t) 
  &approx sum_(j = 0)^(N' - 1) g(t_j) (R_j N_0^((1)) (t_j) + w_j N_0^((2)) (t_j)) \
  $
]