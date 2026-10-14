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
#let remark = thmplain("remark", "Remark")
#set page(margin: 1cm)
#let hk1 = $H^((1))$
#let slp = $cal(S)$
#let dlp = $cal(D)$
#let dlpa = $cal(D)^*$
#let tlp = $cal(T)$
#let sl = $op("SL")$
#let dl = $op("DL")$
#let uin = $u_"in"$
#let dp(x, y) = $lr(chevron.l #x, #y chevron.r)$
#let ip(x, y) = $lr(( #x, #y ))$
#let hd = $hat(h)$
#let arginf = $op("arginf")$
#let c2pi = $C_(2 pi)$
#let jr = $hat(J)$
#let hdj = $grad^((H)) J$
#let hdh = $h^((H))$
#let outomega = $RR^2 without overline(Omega)$
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

#definition[
  Let $(Omega_i)_(i in I)$ be bounded open sets satisfying $forall i, j in I. overline(Omega_i) inter overline(Omega_j) = emptyset$ with connected $C^2$ boundaries $(Gamma_i)_(i in I)$.
  Let $Omega := union_(i in I) Omega_i$, $Gamma := union_(i in I) Gamma_i$.
  Let $k > 0$, let $u in C^1 (RR^2 without Omega)$ satisfy
  $
  cases(
    (laplacian + k^2) u = 0 "in" outomega,
    u = -uin "on" Gamma,
    "SRC" "on" SS^1
  )
  $
]

#definition[
  Let
  $
    G(x, y) := i/4 hk1_0 (k abs(x - y)).
  $
  Let $Omega$ be a bounded open set which has connected $C^2$ boundary $Gamma$. Let
  $
     slp: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma G(x, y) phi(y) dd(y) \
     dlp_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(y) dot grad_y G(x, y) phi(y) dd(y) \
    dlpa_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(x) dot grad_x G(x, y) phi(y) dd(y) \
     tlp_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral.dash_Gamma sum_(i,j) n_i (x) n_j (y) pdv(G, x_i, y_j)(x, y) phi(y) dd(y) \
     sl : C(Gamma) -> C(outomega), phi &|-> integral_Gamma G(x, y) phi(y) dd(y) \
     dl_Gamma : C(Gamma) -> C(outomega), phi &|-> integral_Gamma n(y) dot grad_y G(x, y) phi(y) dd(y)
  $
]
#definition[
  Let $c2pi^k (KK) := C^k (RR \/ 2 pi, KK)$, $x in c2pi^k (RR^2)$, $Gamma_x := {x(t) | t in [0, 2pi)}$.
  Let
  $
    G(x, y) := i/4 hk1_0 (k abs(x - y)),
  $
  $
    slp_x&: c2pi -> c2pi,  phi |-> t |-> integral_0^(2 pi) G(x(t), x(tau)) phi(tau) abs(x'(tau)) dd(tau), \
    dlp_x&: c2pi -> c2pi, phi |-> t |-> integral_0^(2 pi) grad_1 G(x(t), x(tau)) dot (x'_2 (tau), -x'_1 (tau)) phi(y) dd(s(y)).
  $
]
#theorem[
  Let $g_x (t) := -uin(x(t))$, then
  $
  forall eta > 0. exists phi in c2pi. (I/2 + dlp_x - i eta slp_x) phi = g_x and forall x in RR^d without overline(Omega). u(x) = (dl_x - i eta sl_x) phi(x)
  $
]

#definition[Frechet derivative][
  Let $X, Y$ $KK$-norm spaces.
  Let $O subset.eq X$ be open.
  Let $F: O -> Y$, $x in X$.
  A bounded linear operator $D F (x)$ is called the Frechet derivative of $F$ at $x$ if $lim_(h -> 0) (norm(F(x + h) - F(x) - D F (x) [h])_Y) / (norm(h)_X) = 0$.
]
Let $x(t) := (x_1 (t), x_2 (t)), x_1, x_2 in C^2_(2 pi)$ satisfies $forall t in [0, 2 pi). abs(x'(t)) != 0$, counterclockwise and does not intersect itself. Let $Gamma := {x(t) | t in [0, 2 pi)}$.

#definition[Counterclockwise outward vectors][
  $
    n(t) := (n^*(t))/ abs(x'(t)), quad n^*(t) := (x'_2(t), -x'_1(t))
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
#let sl = $op(cal("SL"))$
#let dl = $op(cal("DL"))$
#let slk = $op("SL")$
#let dlk = $op("DL")$
#theorem[
  Let $slk, dlk$ kernels of $sl_Gamma, dl_Gamma$ respectively. Then,
  $
    slk (x, tau) & := G(x, x(tau)) abs(x'(tau)) \
                 & = i/4 hk1_0 (k abs(x - x(tau))) abs(x'(tau)) \
    dlk (x, tau) & := n(tau) dot grad_y G(x, x(tau)) abs(x'(tau)) \
                 & = n^* (tau) dot (i k)/4 (hk1_1 (k abs(x - x(tau))))/(abs(x - x(tau))) (x - x(tau))
  $
  Their shape derivatives are
  $
    (slk)'[h](x, tau) & = (i k)/4 hk1_1 (k abs(x - x(tau))) (x - x(tau)) dot h(tau) / abs(x - x(tau)) abs(x'(tau)) \
                      & quad + i/4 hk1_0 (k abs(x - x(tau))) (x'(tau) dot h'(tau)) / abs(x'(tau)) \
    (dlk)'[h](x, tau) & = (i k)/4 ((n^*)'[h](tau) dot (x - x(tau)) - n^* (tau) dot h(tau)) \
                      & quad times (hk1_1 (k abs(x - x(tau))))/(abs(x - x(tau))) \
                      & quad + (i k^2)/4 (n^* (tau) dot (x - x(tau))) ((x - x(tau)) dot h(tau)) \
                      & quad times (hk1_2 (k abs(x - x(tau)))) / abs(x - x(tau))^2
  $
]
#proof[
  Since $x in.not Gamma$, the kernels are smooth in $tau$, so the chain rule applies directly.
  For $slk$, use $dv(, z) hk1_0 (z) = - hk1_1 (z)$:
  $
    (slk)'[h](x, tau) & = - i/4 k hk1_1 (k r) r'[h] abs(x'(tau)) + i/4 hk1_0 (k r) (abs(x'))'[h](tau)
  $
  where $r := abs(x - x(tau))$, $r'[h] = - (x - x(tau)) dot h(tau) / r$ and $(abs(x'))'[h](tau) = (x'(tau) dot h'(tau)) / abs(x'(tau))$.

  For $dlk$, write $dlk (x, tau) = n^* (tau) dot (i k)/4 f(r) (x - x(tau))$ with $f(r) := hk1_1 (k r) / r$. Then
  $
    (dlk)'[h](x, tau) & = (n^*)'[h](tau) dot (i k)/4 f(r) (x - x(tau)) \
                      & quad + n^* (tau) dot (i k)/4 f'(r) r'[h] (x - x(tau)) \
                      & quad + n^* (tau) dot (i k)/4 f(r) (- h(tau))
  $
  Since $f'(r) = - k hk1_2 (k r) / r$ (from $dv(, z) z^(-1) hk1_1 (z) = - z^(-1) hk1_2 (z)$), substituting gives the result.
]


= Optimization under Boundary integral equation @colton_inverse_2019 @matsushima_2023

#definition[Sesquilinear form][
  $X, Y$: $KK$-norm spaces. $dp(x, y)$ is called a sesquilinear form if
  it is linear in $x$ and conjugate-linear in $y$.
  $dp$ is called non-degenerate if $forall x in X. (forall y in Y. dp(x, y) = 0) ==> x = 0$ and $forall y in Y. (forall x in X. dp(x, y) = 0) ==> y = 0$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any non-degenerate sesquilinear form on $c2pi(CC), c2pi(CC)$. Let $A^*$ adjoint operator of $A$ with respect to $dp(dot, dot)$.
  Let $k >= 2$.
  Let $x in c2pi^k (RR^2), g: c2pi^k (RR^2) -> c2pi^k (CC)$.
  Let $J: c2pi^k (RR^2) times c2pi (CC) -> RR$ Frechet differentiable. Let density $phi_x in c2pi^k$ satisfy the boundary integral equation

  $
    (I/2 + dlp_x - i eta slp_x) phi_x = g_x quad t in [0, 2pi)
  $

  Let $jr(x) := J(x, phi_x)$.
  Assume there exists $grad_phi J(x, phi_x) in c2pi (CC)$ such that for any $h in c2pi (CC), D_phi J (x, phi_x) [h] = Re dp(grad_phi J (x, phi_x), h)$.
  Then $D_x jr(x) [h]$ is given by

  $
    D_x jr(x) [h] & = D_x J(x, phi_x) [h] + Re dp(psi_x, D_x dlp_x [h] phi_x - i eta D_x slp_x [h] phi_x - D_x g_x [h])
  $
  where $psi_x in c2pi (CC)$ satisfies the following adjoint equation:
  $
    (I/2 + dlp_x - i eta slp_x)^* psi_x = - grad_phi J (x, phi_x)
  $
]
#proof[
  Let $L: c2pi^k (RR^2) times c2pi (CC) times c2pi (CC) -> RR$ defined by
  $
    L(x, phi, psi) := J(x, phi) + Re dp(psi, (I/2 + dlp_x - i eta slp_x) phi - g_x)
  $
  Then
  $
    D_x jr(x) [h] & = D_x L(x, phi_x, psi_x) [h] + D_phi L(x, phi_x, psi_x) [D_x phi_x [h]] + D_psi L(x, phi_x, psi_x) [D_x psi_x [h]]
  $
  The first term is
  $
    D_x L(x, phi_x, psi_x) [h] & = D_x J(x, phi_x) [h] + Re dp(psi_x, D_x dlp_x [h] phi_x - i eta D_x slp_x [h] phi_x - D_x g_x [h]) \
  $
  The last two terms vanish since for any $v in c2pi$,
  $
    D_phi L(x, phi, psi_x) [v] & = D_phi J (x, phi) [v] + Re dp(psi_x, (I/2 + dlp_x - i eta slp_x) v) \
                               & = Re dp((I/2 + dlp_x - i eta slp_x)^* psi_x + grad_phi J (x, phi), v) = Re dp(0, v) = 0
  $
  and for any $w in c2pi$,
  $
    D_psi L(x, phi_x, psi) [w] = Re dp(w, (I/2 + dlp_x - i eta slp_x) phi_x - g_x) = Re dp(w, 0) = 0
  $
]
#remark[
  Typically $g_x := - uin compose x$, $J (x, phi) := J(x, (dlp_x - i eta slp_x) phi)$ is used, where $uin$ is the incident wave and $J$ is the objective functional based on shape and scattered field, not density.

  In this case, $D_x g_x (t) [h] = - grad uin(x(t)) dot h(t)$, $grad_phi J (x, phi) = (dlp_x - i eta slp_x)^* grad_u J (x, (dlp_x - i eta slp_x) phi)$, since $dp(grad_phi J, h) = dp(grad_u J, (dlp_x - i eta slp_x) h) = dp((dlp_x - i eta slp_x)^* grad_u J, h)$
]

#remark[
  In the proof above, the step $D_phi L(x, phi_x, psi_x)[D_x phi_x[h]] = 0$ relies on the fact that the shape-induced variation $D_x phi_x[h]$ belongs to $c2pi(CC)$, the test space for which the adjoint equation holds. Here this is trivially satisfied since the boundary spaces map onto themselves smoothly; in more general Sobolev settings, verifying that variations remain valid test functions is a necessary step.
]
#algorithm[
  Assume we have implementation of $J, D_x J, D_phi J, x, x', x'', h, h', h'', slp_x, dlp_x, D_x slp_x, D_x dlp_x, g_x, D_x g_x$.
  + Compute $phi_x$ by solving the boundary integral equation
  + Compute $D_phi J$, then compute $psi_x$ by solving the adjoint equation
  + Compute the Riesz representative $hdj(x)$ of $D_x jr(x)$ to obtain the gradient (e.g. via the spectral coefficients $c'_m, d'_m$)
  + Update the shape: $x_(n + 1) = x_n + lambda hdh$ where $hdh := - hdj(x) / norm(hdj(x))_H$
]
#definition[Hilbertian Regularization][
  Let $X$ be a norm space.
  Let $J: X -> RR$ be Frechet differentiable at $x in X$.
  Let $H subset.eq X$ be a Hilbert space continuously embedded in $X$.
  Since $H$ is a Hilbert space, there exists a Riesz representation $hdj: X -> H$ such that for any $x in X, h in H$,

  $
    ip(hdj(x), h)_H = D J (x) [h] quad forall h in H
  $
  The regularized steepest descent direction $hdh$ is defined as
  $
    hdh := - hdj(x)/norm(hdj(x))_H
  $
]
#theorem[
  The regularized steepest descent direction $hdh$ is the steepest descent direction with respect to $norm(dot)_H$, i.e. $hdh = arginf_(norm(h)_H = 1) D J (x) [h]$.
]
#proof[
  By the Cauchy–Schwarz inequality,
  $
    D J (x) [h] = ip(hdj(x), h)_H >= -norm(hdj(x))_H norm(h)_H = -norm(hdj(x))_H
  $
  for any $norm(h)_H = 1$, with equality if and only if $h = -hdj(x) / norm(hdj(x))_H$.
  Hence
  $
    D J (x) [hdh] = D J (x) [-hdj(x)/norm(hdj(x))_H] = -norm(hdj(x))_H = inf_(norm(h)_H = 1) D J (x) [h]
  $
]
#let h2pi = $H_(2 pi)$
#definition[
  Let $alpha > 0$.
  Let $a_m (phi) := 1/pi integral_0^(2 pi) phi(t) cos(m t) dd(t)$, $b_m (phi) := 1/pi integral_0^(2 pi) phi(t) sin(m t) dd(t)$.
  Let $ip(phi, psi)_h2pi^k := 1/2 a_0 (phi) a_0 (psi) + sum_(m = 1)^infinity (1 + alpha m^2)^k (a_m (phi) a_m (psi) + b_m (phi) b_m (psi))$.
  Let $h2pi^k (RR) := {a_0 / 2 + sum_(m = 1)^infinity (a_m cos(m t) + b_m sin(m t)) | a_m, b_m in RR, a_0^2 + sum_(m = 1)^infinity (1 + alpha m^2)^k (a_m^2 + b_m^2) < infinity}$.
  $(h2pi^k (RR), ip(dot, dot)_h2pi^k)$ is a Hilbert space.
]

$h2pi^3 (RR) subset.neq c2pi^2 (RR)$ may be used for regularization.
#let hdr = $h^((R_N))$
#let hdk = $h^((h2pi^k))$
#definition[
  Let $R_N := {a_0 / 2 + sum_(m = 1)^(N - 1) (a_m cos(m t) + b_m sin(m t)) | a_m, b_m in RR} subset.neq h2pi^k (RR)$.
  $R_N$ is continuously embedded in $h2pi^k (RR)$.
]

#theorem[
  The coefficients ${c_m}_(m = 0)^(N - 1) union {d_m}_(m = 1)^(N - 1)$ of the finite-dimensional steepest descent direction $hdr$ can be computed by

  $
    c'_m := (D_x J (x) [cos(m t)]) / (1 + alpha m^2)^k, quad d'_m := (D_x J (x) [sin(m t)]) / (1 + alpha m^2)^k
  $

  $
    S := 1/2 c'_0^2 + sum_(m = 1)^(N - 1) (1 + alpha m^2)^k (c'_m^2 + d'_m^2), quad c_m := c'_m / sqrt(S), quad d_m := d'_m / sqrt(S)
  $
  where $c'_m, d'_m$ are the Fourier coefficients of the unnormalized Riesz representation $hdj(x)$.
]

#theorem[Error estimate][
  Let $g := hdj(x)$ be the unnormalized Riesz representation in $h2pi^k(RR)$ and $g_N$ its truncation in $R_N$. If $g in h2pi^(k + s)(RR)$ for some $s > 0$, then
  $
    norm(g - g_N)_(h2pi^k) <= (1 + alpha N^2)^(-s/2) norm(g)_(h2pi^(k + s))
  $
]
#proof[
  Let $c'_m, d'_m$ be the Fourier coefficients of $g$ as defined above.
  The squared norm of the truncation error is the tail of the series:
  $
    norm(g - g_N)_(h2pi^k)^2 & = sum_(m = N)^infinity (1 + alpha m^2)^k ((c'_m)^2 + (d'_m)^2) \
                             & = sum_(m = N)^infinity (1 + alpha m^2)^(-s) (1 + alpha m^2)^(k + s) ((c'_m)^2 + (d'_m)^2) \
                             & <= (1 + alpha N^2)^(-s) sum_(m = N)^infinity (1 + alpha m^2)^(k + s) ((c'_m)^2 + (d'_m)^2) \
                             & <= (1 + alpha N^2)^(-s) norm(g)_(h2pi^(k + s))^2
  $
  Taking square roots yields the claimed bound.
]
#let ju = $tilde(J)$
#theorem[Riesz representation for point evaluation][
  Let $x_0 in RR^2 without overline(Omega_x)$.
  Let $ju in C^1 (RR^2, RR)$
  Let $J(x, phi) := ju(Re u_phi (x), Im u_phi (x)), u_phi (x) := ((alpha dlp_x - i eta slp_x) phi)$.
  (If $c in CC$ and $ju(u) = abs(u - c)^2 = (Re u - Re c)^2 + (Im u - Im c)^2$, then $(pdv(ju, x_1) + i pdv(ju, x_2))(u) = 2 (u - c)$.)
  Then, the Riesz representation of the Frechet derivative of $J$ with respect to $phi$ under the sesquilinear form
  $
    dp(f, g) := integral_0^(2 pi) f(t) overline(g(t)) dd(t),
  $
  is given by
  $
    grad_phi J(x, phi)(tau) =
    (pdv(ju, x_1) + i pdv(ju, x_2)) (u_phi (x_0)) overline(K(x_0, tau)),
  $
  where $pdv(ju, x_1), pdv(ju, x_2)$ are the partial derivatives of $ju$ with respect to its first and second arguments, and
  $
    K(x_0, tau) := alpha tilde(D)(x_0, tau) - i eta tilde(S)(x_0, tau),
  $
  with
  $
    tilde(S)(x_0, tau) := G(x_0, x(tau)) abs(x'(tau)),
    tilde(D)(x_0, tau) := n(tau) dot grad_y G(x_0, x(tau)) abs(x'(tau))
  $
  the kernels of $sl_x$, $dl_x$ with jacobian multiplied, evaluated at $x_0$.
]
#proof[
  Let $u := u_phi (x_0)$ and $K(x_0, tau) := alpha tilde(D)(x_0, tau) - i eta tilde(S)(x_0, tau)$.
  Since
  $
    D_phi J(x, phi)[h] = D ju(u) [(alpha dlp_x - i eta slp_x) h (x_0)]
  $
  and expanding the evaluation operators gives
  $
    (alpha dlp_x - i eta slp_x) h (x_0)
    = integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau),
  $
  using that $ju$ is real-valued, hence $D ju$ is real-linear,
  $
    D_phi J(x, phi)[h] & = pdv(ju, x_1)(u) Re integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau) + pdv(ju, x_2)(u) Im integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau) \
                       & = Re integral_0^(2 pi) (pdv(ju, x_1) - i pdv(ju, x_2))(u) K(x_0, tau) h(tau) dd(tau) \
                       & = Re integral_0^(2 pi) (pdv(ju, x_1) + i pdv(ju, x_2))(u) overline(K(x_0, tau)) overline(h(tau)) dd(tau),
  $
  where the last equality uses $Re(z) = Re(overline(z))$.
  Comparing with $Re dp(grad_phi J, h)$ gives the claimed representation.
]

#bibliography("main.bib")
