#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)
#let definition = thmplain("definition", "Definition")
#let theorem = thmplain("theorem", "Theorem")
#let proof = thmproof("proof", "Proof")
#let span = $op("span")$
#let sign = $op("sign")$

== Basic Properties

#theorem[
$
integral_0^(2 pi) e^(i m x) dd(x) = cases(2 pi &(m = 0), 0 &(m in ZZ without {0})) #<fourier-integral> \
"p.v." integral_0^(2 pi) cot(x/2) e^(i m x) dd(x) = 2 pi i sign(m) quad m in ZZ without {0} #<cot-integral> \
integral_0^(2 pi) log(4 sin^2 x/2) e^(i m x) dd(x) = cases(0 &(m = 0), -(2 pi)/(abs(m)) &(m in ZZ without {0})) #<log-integral>
$
]
#theorem[
  $forall N' in NN. t_j := (2 pi j)/N'. forall m in ZZ.$
  $
  (2 pi)/N' sum_(j=0)^(N'-1) e^(i m t_j) = cases(2 pi &(m equiv 0 mod N'), 0 &("otherwise"))
  $
] <fourier-sum>

== Subspace $U_N$

#definition[
  $U_N := span({e^(i m x) | m in ZZ, abs(m) < N})$
]
#theorem[
  $forall N in NN. N$-point trapezoidal rule is exact for $U_N$.
]
#theorem[
  Let $N' := dim U_N = 2 N - 1$.
  Let $t_j := (2 pi j)/N'$ for $j = 0, ..., N' - 1$.
  $forall f, g in U_N.$
  $
  integral_0^(2 pi) f(t) g(t) dd(t) = (2 pi)/N' sum_(j=0)^(N'-1) f(t_j)
  $
]
#proof[
  $f(t) g(t) in U_(2 N - 1)$
]

== Quadratures

#theorem[
$forall N in NN. forall f in U_N.$
$
f(x) &= sum_(abs(m) < N) integral_0^(2 pi) f(t) e^(-i m t)/sqrt(2 pi) dd(t) dot e^(i m x)/sqrt(2 pi) \
&= 1/(2 pi) sum_(abs(m) < N) (2 pi)/N' sum_(j=0)^(N'-1) f(t_j) e^(-i m t_j) dot e^(i m x) \
&= sum_(j = 0)^(N'-1) f(t_j) dot ((1/N' sum_(abs(m) < N) e^(- i m t_j)) e^(i m x))
$
]

Assume $f = P_n f$, then

$
integral_0^(2 pi) cot(x/2) f'(x) d(x) 
&=_("Fourier Expansion\nTrapezoidal Rule") integral_0^(2 pi) cot(x/2) dv(,x)(1/(2π) sum_(0 <= abs(m) <= n) pi/n sum_(j=0)^(2n-1) f(t_j) e^(-i m t_j) e^(i m x)) dd(x) \
&= integral_0^(2 pi) cot(x/2) (1/(2π) sum_(1 <= abs(m) <= n) pi/n sum_(j=0)^(2n-1) f(t_j) e^(-i m t_j) i m e^(i m x)) dd(x) \
&= 1/(2π) sum_(1 <= abs(m) <= n) pi/n sum_(j=0)^(n-1) f(t_j) e^(-i m t_j) i m integral_0^(2 pi) cot(x/2)  e^(i m x) dd(x) \
&=_(#ref(<cot-integral>)) 1/(2π) sum_(1 <= abs(m) <= n) pi/n sum_(j=0)^(n-1) f(t_j) e^(-i m t_j) i m dot 2 pi i sign(m) \
&= sum_(j =0)^(n-1) f(t_j) dot (- pi/n sum_(1 <= abs(m) <= n) m sign(m) e^(-i m t_j)) \
&= sum_(j =0)^(n-1) f(t_j) dot ((2 pi)/n sum_(1 <= m <= n) m sin(m t_j)) \
$