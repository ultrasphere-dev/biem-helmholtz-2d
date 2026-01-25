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

== Integrals involving Fourier basis functions

The below thorem is used everywhere.

#theorem[Fourier Integral][
  $
    integral_0^(2 pi) e^(i m x) dd(x) = cases(2 pi &(m = 0), 0 &(m in ZZ without {0})) #<fourier-integral>
$
]
#proof[
  $
    integral_0^(2 pi) e^(i m x) dd(x) & = (e^(i m dot 2 pi) - e^(i m dot 0))/(i m)  \
                                   & = cases(2 pi &(m = 0), 0 &(m in ZZ without {0}))
  $
]

Integral equations coming from common partial differential equations in $RR^2$ often involve singular integrals, derivative of integrals.

#definition[Cauchy Principle value on a circle][
  Let $alpha in (0, 1], f in C^(0, alpha) (0, 2 pi)$
  $
  p.v. integral_0^(2 pi) f(x) cot(x/2) dd(x) := lim_(epsilon -> 0) (integral_(epsilon)^(2 pi - epsilon) f(x) cot(x/2) dd(x))
  $
]
#definition[Hadamard Finite-Part on a circle][
  Let $alpha in (0, 1], n in NN without {1}, f in C^(n,alpha) (0, 2 pi)$
  $
  p.f. integral_0^(2 pi) f(x) cot^n (x/2) dd(x) := lim_(epsilon -> 0) F_0(epsilon)
  $
  where
  $
  forall epsilon in (0, pi). integral_(epsilon)^(2 pi - epsilon) f(x) cot^n (x/2) dd(x) = F_0 (epsilon) + F_1 (epsilon) \
  lim_(epsilon -> 0) F_0 (epsilon) < infinity \
  lim_(epsilon -> 0) log(4 sin^2 (epsilon/2)) 1/(F_1 (epsilon)) < infinity
  $
]

#lemma[
  $
    p.v. integral_0^(2 pi) cot(x/2) e^(i m x) dd(x) = 2 pi i sign(m) quad m in ZZ without {0} #<cot-integral> \
    integral_0^(2 pi) log(4 sin^2 x/2) e^(i m x) dd(x) = cases(0 &(m = 0), -(2 pi)/(abs(m)) &(m in ZZ without {0})) #<log-integral>
  $
]
#lemma[
  $forall N' in NN. t_j := (2 pi j)/N'. forall m in ZZ.$
  $
    (2 pi)/N' sum_(j=0)^(N'-1) e^(i m t_j) = cases(2 pi &(m equiv 0 mod N'), 0 &("otherwise"))
  $
] <fourier-sum>

== Subspace $U_N$

#definition[
  $U_N := span({e^(i m x) | m in ZZ, abs(m) < N})$
]
#theorem[Trapezoidal Rule for $U_N$][
  $forall N in NN. N$-point trapezoidal rule is exact for $U_N$.
] <trapezoidal-rule>
#proof[
  @fourier-sum
]
#lemma[Trapezoidal Rule for inner product for $U_N$][
  Let $N' := dim U_N = 2 N - 1$.
  Let $t_j := (2 pi j)/N'$ for $j = 0, ..., N' - 1$.
  $forall f, g in U_N.$
  $
    integral_0^(2 pi) f(t) g(t) dd(t) = (2 pi)/N' sum_(j=0)^(N'-1) f(t_j)
  $
] <dft-trapezoidal>
#proof[
  Since, $f(t) g(t) in U_(2 N - 1)$ @trapezoidal-rule can be applied with $N' = 2 N - 1$ points.
]

== Quadratures

#lemma[
  $forall N in NN. forall f in U_N. N := 2 N - 1. t_j := (2 pi j)/N'.$
  $
    f(x) = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) e^(i m x))
  $
] <dft>
#proof[
  $
    f(x) & = sum_(abs(m) < N) integral_0^(2 pi) f(t) e^(-i m t)/sqrt(2 pi) dd(t) dot e^(i m x)/sqrt(2 pi) \
         & =_(because #ref(<dft-trapezoidal>)) 1/(2 pi) sum_(abs(m) < N) (2 pi)/N' sum_(j=0)^(N'-1) f(t_j) e^(-i m t_j) dot e^(i m x) \
         & = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) e^(i m x))
  $
]

#theorem[Kussmaul--Martensen (Kress) quadrature for $U_N$][
  $forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'.$
  $
    integral_0^(2 pi) log(4 sin^2 t/2) f(t) dd(t) = sum_(j=0)^(N'-1) f(t_j) dot (-(4 pi)/N' sum_(m = 1)^(N-1) (cos m t_j)/m)
  $
]
#proof[
  $
    integral_0^(2 pi) log(4 sin^2 t/2) f(t) dd(t) & =_(because #ref(<dft>)) integral_0^(2 pi) log(4 sin^2 t/2) (sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) e^(i m t))) dd(t) \
                                                  & = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) integral_0^(2 pi) log(4 sin^2 t/2) e^(i m t) dd(t)) \
                                                  & = sum_(j=0)^(N'-1) f(t_j) dot (1/N' sum_(0 < abs(m) < N) (-(2 pi)/(abs(m))) e^(- i m t_j)) \
                                                  & = sum_(j=0)^(N'-1) f(t_j) dot (-(4 pi)/N' sum_(m = 1)^(N-1) (cos m t_j)/m)
  $
]

#theorem[Garrick--Wittich quadrature for $U_N$][
  $forall N in NN. forall f in U_N. N' := 2 N - 1. t_j := (2 pi j)/N'.$
  $
    p.v. integral_0^(2 pi) cot(t/2) f'(t) dd(t) = sum_(j = 0)^(N'-1) f(t_j) dot (-(4 pi)/N' sum_(m = 1)^(N-1) m cos(m t_j))
  $
]
#proof[
  $
    p.v. integral_0^(2 pi) cot(t/2) f'(t) dd(t) & =_(because #ref(<dft>)) p.v. integral_0^(2 pi) cot(t/2) (sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) (e^(i m t))')) dd(t) \
                                                & = p.v. integral_0^(2 pi) cot(t/2) (sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(0 < abs(m) < N) e^(- i m t_j) (i m e^(i m t)))) dd(t) \
                                                & = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(0 < abs(m) < N) e^(- i m t_j) (i m space p.v. integral_0^(2 pi) cot(t/2) e^(i m t) dd(t))) \
                                                & = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(0 < abs(m) < N) e^(- i m t_j) (i m dot 2 pi i sign(m))) \
                                                & = sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(0 < abs(m) < N) (- 2 pi abs(m)) e^(- i m t_j)) \
                                                & = sum_(j = 0)^(N'-1) f(t_j) dot (-(4 pi)/N' sum_(m = 1)^(N-1) m cos(m t_j))
  $
]
// #theorem[Trapezoidal Rule for differentiated functions in $U_N$][
//   Let $N' := dim U_N = 2 N - 1$.
//   Let $t_j := (2 pi j)/N'$ for $j = 0, ..., N' - 1$.
//   $forall f, g in U_N.$
// ]
// #proof[
//   $
//   integral_0^(2 pi) f'(t) g(t) dd(t) &= integral_0^(2 pi) g(t) (sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) (e^(i m t))')) dd(t) \
//   &= sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) integral_0^(2 pi) g(t) (i m e^(i m t)) dd(t)) \
//   &=_(because #ref(<dft-trapezoidal>)) sum_(j = 0)^(N'-1) f(t_j) dot (1/N' sum_(abs(m) < N) e^(- i m t_j) (i m (2 pi)/N' sum_(k=0)^(N'-1) g(t_k) e^(i m t_k))) \
//   $
// ]
