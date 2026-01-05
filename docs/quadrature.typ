#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

#let sign = $op("sign")$

$
integral_0^(2 pi) cot(x/2) e^(i m x) dd(x) = 2 pi i sign(m) quad m in ZZ without {0} #<cot-integral> \
integral_0^(2 pi) log abs(4 sin^2 x/2) e^(i m x) dd(x) = cases(0 &(m = 0), -(2 pi)/(abs(m)) &(m in ZZ without {0})) #<log-integral>
$ 

$
integral_0^(2 pi) cot(x/2) f'(x) d(x) 
&=_("Fourier Expansion") integral_0^(2 pi) cot(x/2) dv(,x)(1/(2π) sum_(0 <= abs(m) <= n) integral_0^(2 pi) f(t) e^(-i m t)dd(t) dot e^(i m x)) dd(x) \
&= integral_0^(2 pi) cot(x/2) (1/(2π) sum_(1 <= abs(m) <= n) integral_0^(2 pi) f(t) e^(-i m t)dd(t) dot i m e^(i m x)) dd(x) \
&= 1/(2π) sum_(1 <= abs(m) <= n) integral_0^(2 pi) f(t) e^(-i m t)dd(t) dot i m integral_0^(2 pi) cot(x/2)  e^(i m x) dd(x) \
&=_(#ref(<cot-integral>)) 1/(2π) sum_(1 <= abs(m) <= n) integral_0^(2 pi) f(t) e^(-i m t)dd(t) dot i m dot 2 pi i sign(m) \
&=_("Trapezoidal Rule") 1/(2π) sum_(1 <= abs(m) <= n) pi/n sum_(j=0)^(n-1) f(t_j) e^(-i m t_j) dot i m dot 2 pi i sign(m) \
&= sum_(j =0)^(n-1) f(t_j) dot (- pi/n sum_(1 <= abs(m) <= n) m sign(m) e^(-i m t_j)) \
&= sum_(j =0)^(n-1) f(t_j) dot (2 pi/n sum_(1 <= m <= n) m sin(m t_j)) \
$