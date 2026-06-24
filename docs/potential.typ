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
#set page(margin: 1cm)
#let hk1 = $H^((1))$
= Layer potentials

#theorem[Weakly singular kernels ][
  Let $m in NN, k in NN_0$.
  Let $D subset RR^m$ bounded open domain of $C^(k + 1)$ boundary $Gamma$.
  Let $K: Gamma^2 -> KK$ weakly singular, i.e. continuous on $T := Gamma^2 backslash {(x, x) | x in Gamma}$ and $exists M > 0. exists alpha in (0, m - 1]. forall x, y in T. abs(K(x, y)) <= M abs(x - y)^(alpha - m + 1)$.
  Then $(A phi) (x) := integral_Gamma K(x, y) phi(y) dd(s(y))$ is a compact linear operator on $C^k (Gamma)$.
]
#proof[
  The case $k = 0$ follows #cite(<kress_linear_2014>, supplement: "Theorem 2.29, 2.30").
  For $k >= 1$ this is wrong.
]

Let $x(t) := (x_1 (t), x_2 (t)), x_1, x_2 in C^2_(2 pi)$ satisfies $forall t in [0, 2 pi). abs(x'(t)) != 0$, counterclockwise and does not intersect itself. Let $Gamma := {x(t) | t in [0, 2 pi)}$.

#let slp = $cal(S)$
#let dlp = $cal(D)$
#let dlpa = $cal(D)^*$
#let tlp = $cal(T)$
#definition[Counterclockwise outward vectors][
  $
    n(t) := (n^*(t))/ abs(x'(t)), quad n^*(t) := (x'_2(t), -x'_1(t))
  $
]
#definition[
  $
    G(x, y) := i/4 hk1_0 (k abs(x - y))
  $
  $
     slp_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma G(x, y) phi(y) dd(y) \
     dlp_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(y) dot grad_y G(x, y) phi(y) dd(y) \
    dlpa_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(x) dot grad_x G(x, y) phi(y) dd(y) \
     tlp_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral.dash_Gamma sum_(i,j) n_i (x) n_j (y) pdv(G, x_i, y_j)(x, y) phi(y) dd(y)
  $
]
#theorem[@zhang_superconvergence_2010][
  $
                                      grad_y G(x, y) & = (i k)/4 (hk1_1 (k abs(x - y)))/(abs(x - y)) (x - y) \
    sum_(i,j) n_i (x) n_j (y) pdv(G, x_i, y_j)(x, y) & = (i k)/4 hk1_0 (k abs(x - y)) (k n(x) dot (x - y) n(y) dot (x - y)) / (abs(x - y)^2) \
                                                     & + (i k)/4 hk1_1 (k abs(x - y)) ((n(x) dot n(y))/(abs(x - y)) - 2 (n(x) dot (x - y) n(y) dot (x - y))/(abs(x - y)^3))
  $
]
#let ht1 = $cal(H)^((1))$
#definition[
  $ht1_n (x) := x^n hk1_n (x)$
]
#let dx = $op("dx")$
#let dxa = $op("dxa")$
#let dh = $op("dh")$
#let ntaudotdxdxa2 = $op(A_1)$
#definition[
  $
     dx(t, tau) & := x(t) - x(tau) \
    dxa(t, tau) & := abs(dx(t, tau))
  $
]
#theorem[
  Let $S$, $D$ kernels of $slp_Gamma$, $dlp_Gamma$ respectively.
  $
    S(t, tau) & := G(x(t), x(tau)) abs(x'(tau)) \
              & = i/4 hk1_0 (k abs(x(t) - x(tau))) abs(x'(tau)) \
              & = i/4 hk1_0 (k dxa(t, tau)) abs(x'(tau)) \
    D(t, tau) & := n(tau) dot grad_y G(x(t), x(tau)) abs(x'(tau)) \
              & = n^* (tau) dot (i k)/4 (hk1_1 (k dxa(t, tau)))/dxa(t, tau) dx(t, tau) \
              & = i/4 (ht1_1 (k dxa(t, tau))) ntaudotdxdxa2 (t, tau) \
  $
  where
  $
    ntaudotdxdxa2 (t, tau) & := (n^* (tau) dot dx(t, tau))/(dxa(t, tau)^2) \
                           & := ((x'_2(tau), -x'_1(tau)) dot (x(t) - x(tau)))/(abs(x(t) - x(tau))^2) \
                           & ->_(tau -> t) (x''_1(t) x'_2(t) - x''_2(t) x'_1(t)) / (2 abs(x'(t))^2) \
  $
]
#theorem[Circle case][
  $forall m in ZZ.$
  $
                       (slp_(rho SS^1) e^(i m t)) (t) & = (i pi rho)/2 hk1_abs(m) (k rho) J_abs(m) (k rho) e^(i m t) \
    ((I_(rho SS^1)/2 + dlp_(rho SS^1)) e^(i m t)) (t) & = (i pi k rho)/2 hk1_abs(m) (k rho) J'_abs(m) (k rho) e^(i m t) \
  $
]

== Frechet Derivative
#let define = $<==>_"define"$
#let frd = $op("FRD")$
#definition[Frechet derivative][
  $X, Y$: $KK$-norm spaces, $forall O in cal(O)(X). forall F: O -> Y. F in frd(X) define exists A_x in B(X, Y). lim_(h -> 0) (norm(F(x + h) - F(x) - A_x [h])_Y) / (norm(h)_X) = 0$
]
#theorem[Chain rule][
  $X, Y, Z$: $KK$-norm spaces, $forall x_0 in X. forall F in frd(x_0). y_0 := F(x_0). forall G in frd(y_0). H := G compose F. H'(x_0) = G'(y_0) compose F'(x_0)$
]
#theorem[Frechet derivative of $dx, dxa$][
  $
     dx'[h] & = (x(t) - x(tau))'[h] = dh, quad dh(t, tau) := h(t) - h(tau) \
    dxa'[h] & = (abs(x(t) - x(tau)))'[h] = (dx(t, tau) dot dh(t, tau)) / (dxa(t, tau)) \
  $
]
#let dxapdxa = $op(A_2)$
Since $dxa'[h]$ is not programmatically evaluable at $t = tau$, the following $dxapdxa$ is defined.
#theorem[
  $
    dxapdxa (t, tau) := (dxa'[h])/(dxa) = (dx(t, tau) dot dh(t, tau)) / (dxa(t, tau))^2
    ->_(tau -> t) (x'(t) dot h'(t)) / abs(x'(t))^2
  $
]
#theorem[
  $
    dv(, x) x^(-n) hk1_n (x) = - x^(-n) hk1_(n + 1) (x)
  $
]
Shape derivatives of $slp$ and $dlp$ may be expressed by $ht1 (f(z)), dxapdxa$ as below, making it possible to evaluate the limit value $t = tau$ programmatically.

#theorem[Shape Derivative of $slp$][
  Let $S$ kernel of $slp_Gamma$.
  $
      S (t, tau) & = i/4 S_1 (t, tau) \
    S_1 (t, tau) & := hk1_0 (k dxa(t, tau)) abs(x'(tau)) \
  $
  Then
  $
    (S_1)'[h](t, tau) & = pdv(S_1, dxa) dxa'[h](t, tau) + pdv(S_1, abs(x')) abs(x')'[h](tau) \
                      & = - k hk1_1 (k dxa(t, tau)) dxa'[h](t, tau) abs(x'(tau)) + hk1_0 (k dxa(t, tau)) (x'(tau) dot h'(tau)) / (abs(x'(tau))) \
                      & = - ht1_1 (k dxa(t, tau)) dxapdxa (t, tau) abs(x'(tau)) + hk1_0 (k dxa(t, tau)) (x'(tau) dot h'(tau)) / (abs(x'(tau))) \
        S'[h](t, tau) & = i/4 (S_1)'[h](t, tau) \
  $
]
#theorem[Shape Derivative of $dlp$][
  Let $D$ kernel of $dlp_Gamma$.
  $
      D (t, tau) & := (i k^2)/4 D_1 (t, tau) \
    D_1 (t, tau) & := D_2 (t, tau) D_3 (t, tau) \
    D_2 (t, tau) & := (k dxa(t, tau))^(-1) hk1_1 (k dxa(t, tau)) \
    D_3 (t, tau) & := n^*(tau) dot dx (t, tau)
  $
  Then
  $
    (n^*)'[h](t) & = (h'_2(t), -h'_1(t)) \
  $
  $
    (D_2)'[h](t, tau) & = dxa' [h] pdv(D_2, dxa) \
                      & = - k dxa' [h](t, tau) (k dxa(t, tau))^(-1) hk1_2 (k dxa(t, tau)) \
                      & = - ht1_2 (k dxa(t, tau)) dxapdxa (t, tau) (k dxa(t, tau))^(-2) \
                      \
    (D_3)'[h](t, tau) & = dh (t, tau) dot n^*(tau) + dx (t, tau) dot (n^*)'[h](tau) = D_4 (t, tau) (dxa (t, tau))^2 \
    (D_1)'[h](t, tau) & = (D_2)'[h](t, tau) D_3 (t, tau) + D_2 (t, tau) (D_3)'[h](t, tau) \
                      & = - k^(-2) ht1_2 (k dxa(t, tau)) dxapdxa (t, tau) ntaudotdxdxa2 (t, tau) \
                      & + k^(-2) ht1_1 (k dxa(t, tau)) D_4 (t, tau) \
        D'[h](t, tau) & = (i k^2)/4 (D_1)'[h](t, tau) \
                      & = i/4(- ht1_2 (k dxa(t, tau)) dxapdxa (t, tau) ntaudotdxdxa2 (t, tau) + ht1_1 (k dxa(t, tau)) D_4 (t, tau))
  $
  where
  $
    D_4 (t, tau) & := ((n^* (tau) dot dh(t, tau)) + ((n^*)'[h](tau) dot dx(t, tau))) / dxa(t, tau)^2 \
                 & ->_(tau -> t) ((h'_2(t) x''_1(t) - h'_1(t) x''_2(t)) + (x'_2(t) h''_1(t) - x'_1(t) h''_2(t))) / (2 abs(x'(t))^2)
  $
]


#bibliography("neumann.bib")
