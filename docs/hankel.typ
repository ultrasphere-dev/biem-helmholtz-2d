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

= Integration of $N_0, hk_0$

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
  z^k C[[z]] &:= {z^k sum_(n = 0)^infinity a_n z^n | forall n in NN. a_n in CC} \
  e^k C[[e]] &:= {f(e^(i z)) | f(z) in z^k CC[[z]]}
  $
]
#lemma[
  $
  J_0 (z) &= 1 + z CC[[z]] \
  N_0 (z) &= 2/pi (log z/2 + C) J_0 (z) + z CC[[z]]
  $
]
#lemma[
  $
  forall f in C[[e]] without e C[[e]]. log f(z) in C[[e]] ?
  $
]
#let n01 = $N_0^((1,f))$
#let n02 = $N_0^((2,f))$
#lemma[
  Let $f in e CC[[e]] without e^2 C[[e]]$.
  $
  N_0 (f(z)) &= 1/pi log(4 sin^2 z/2) J_0 (f(z)) + 2/pi (log f(z)/2 - log 2 sin z/2 + C) J_0 (f(z)) + z CC[[z]] \
  &= 1/pi log(4 sin^2 z/2) J_0 (f(z)) + 2/pi (log f(z)/(4sin z/2) + C) J_0 (f(z)) + z CC[[z]] \
  &= n01 (z) log (4 sin^2 z/2) + n02 (z) \
  $
  where $n01, n02 in e C[[e]]$ and 
  $
  n01 (z) &= 1/pi J_0 (f(z)) #<n01> \
  n02 (z) &= N_0 (f(z)) - n01 (z) log (4 sin^2 z/2) quad (z != 0) #<n02-not0>
  $
  
  Note that analycity of $n02$ at $0$ is not obvious but by L'Hospital's rule we have
  $
  lim_(z -> 0) n02 (z) = 2/pi (log (f'(0))/2 + C) #<n02-0>
  $
]
#theorem[Integral of $hk_0$][
  Let $f in e C[[e]] without e^2 C[[e]], g in C [[e]]$.
  $
  integral_0^(2 pi) g(t) N_0 (f(t)) dd(t) 
  &approx sum_(j = 0)^(N' - 1) g(t_j) (R_j n01 (t_j) + w_j n02 (t_j)) \
  $
]
Implementation: assume we have implementation of $J_0, N_0, f, f'$.
+ Compute ${N_0 (f(t_j))}_(j = 0)^(N' - 1)$ directly ($N_0 (f(t_0))$ will be `NaN`).
+ Compute ${n01 (t_j)}_(j = 0)^(N' - 1)$ using @n01.
+ Compute ${n02 (t_j)}_(j = 0)^(N' - 1)$using @n02-not0  ($n02 (t_0)$ will be temporarily `NaN`).
+ Replace $n02 (t_0)$ with the value from @n02-0.
+ Compute the sum.
#lemma[
  $
  dv(,t) (log 4 sin^2 (t/2)) = cot(t/2)
  $
]
#lemma[
  $
  J'_0 (z) &= 0 + z CC[[z]] \
  N'_0 (f(z)) &= n01 (z) cot(z/2) + n01' (z) log(4 sin^2 z/2) + n02' (z) \
  $
  where
  $
  n01' (z) &= 1/pi J'_0 (f(z)) f'(z) \
  n02' (z) &= N'_0 (f(z)) - n01' (z) log (4 sin^2 z/2) \
  $
  Note that analycity of $n02'$ at $0$ is not obvious but by L'Hospital's rule we have
  $
  lim_(z -> 0) n02' (z) &= lim_(z -> 0) 2/pi ((f'(z))/f(z) -1/2 cot(z/2)) J_0 (f(z)) \
  &= 1/pi lim_(z -> 0)  (2 f'(z) - f(z) cot(z/2))/f(z) \
  &= 1/pi lim_(z -> 0)  (2 f''(z) - f'(z) cot(z/2) + (1/2) f(z) sin^(-2) (z/2))/(f'(z)) \
  &= (f''(0))/(pi f'(0)) #<n02d-0>
  $
]
#theorem[
  Let $f in e CC[[e]], g in C[[e]], h in e C[[e]]$.
  $
  p.v. integral_0^(2 pi) g(t) N_0' (f(t)) dd(t) 
  &approx sum_(j = 0)^(N' - 1) g(t_j) (R_j' n01 (t_j) + w_j n02' (t_j)) \
  $
]