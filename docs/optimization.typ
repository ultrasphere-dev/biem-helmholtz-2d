#import "@preview/equate:0.3.2": *
#import "@preview/physica:0.9.7": *
#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)
#let definition = thmplain("definition", "Definition")
#let theorem = thmplain("theorem", "Theorem")
#let algorithm = thmplain("algorithm", "Algorithm")
#let remark = thmplain("remark", "Remark")
#let lemma = thmplain("lemma", "Lemma")
#let proof = thmproof("proof", "Proof")
#set page(margin: 4mm, paper: "jis-b5")
#let hk1 = $H^((1))$
#let sl = $op("SL")$
#let dl = $op("DL")$
#let slp = $cal(S)$
#let dlp = $cal(D)$
#let dlpa = $cal(D)^*$
#let tlp = $cal(T)$
#let uin = $u_"in"$
#let dp(x, y) = $lr(chevron.l #x, #y chevron.r)$
#let ip(x, y) = $lr(( #x, #y ))$
#let jp = $tilde(J)$
#let hd = $hat(h)$
#let arginf = $op("arginf")$
#let c2pi = $C_(2 pi)$
#let jr = $hat(J)$
#let hdj = $grad^((H)) J$
#let hdh = $h^((H))$
= Optimization under Boundary integral equation @colton_inverse_2019 @matsushima_2023

#definition[
  Let $c2pi^k (KK) := C^k (RR \/ 2 pi, KK)$, $x in c2pi^k (RR^2)$, $Gamma_x := {x(t) | t in [0, 2pi)}$.
  $
    slp_x phi (x) := integral_Gamma_x G(x, y) phi(y) dd(s(y)), quad dlp_x phi (x) := integral_Gamma_x pdv(G(x, y), n(y)) phi(y) dd(s(y)), quad G(x, y) := i/4 hk1_0 (k abs(x - y))
  $
]
#definition[Frechet derivative][
  Let $X, Y$ $KK$-norm spaces.
  Let $O subset.eq X$ be open.
  Let $F: O -> Y$, $x in X$.
  A bounded linear operator $D F (x)$ is called the Frechet derivative of $F$ at $x$ if $lim_(h -> 0) (norm(F(x + h) - F(x) - D F (x) [h])_Y) / (norm(h)_X) = 0$.
]
#definition[Sesquilinear form][
  $X, Y$: $KK$-norm spaces. $dp(x, y)$ is called a sesquilinear form if
  it is linear in $x$ and conjugate-linear in $y$.
  $dp$ is called non-degenerate if $forall x in X. (forall y in Y. dp(x, y) = 0) ==> x = 0$ and $forall y in Y. (forall x in X. dp(x, y) = 0) ==> y = 0$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any non-degenerate sesquilinear form on $c2pi(CC), c2pi(CC)$. Let $A^*$ adjoint operator of $A$ with respect to $dp(dot, dot)$.
  Let $k >= 2$.
  Let $x in c2pi^k (RR^2), g: c2pi^k (RR^2) -> c2pi^k (CC)$.
  Let $jp: c2pi^k (RR^2) times c2pi (CC) -> RR$ Frechet differentiable. Let density $phi_x in c2pi^k$ satisfy the boundary integral equation

  $
    (I/2 + dlp_x - i eta slp_x) phi_x = g_x quad t in [0, 2pi)
  $

  Let $jr(x) := J(x, phi_x)$.
  Assume there exists $grad_phi J(x, phi_x) in c2pi (RR)$ such that for any $h in c2pi (RR), D_phi jp (x, phi_x) [h] = dp(grad_phi jp (x, phi_x), h)$.
  Then $D_x jr(x) [h]$ is given by

  $
    D_x jr(x) [h] & = D_x jp(x, phi_x) [h] + Re dp(psi_x, D_x dlp_x [h] phi_x - i eta D_x slp_x [h] phi_x - D_x g_x [h])
  $
  where $psi_x in c2pi (CC)$ satisfies the following adjoint equation:
  $
    (I/2 + dlp_x - i eta slp_x)^* psi_x = - grad_phi jp (x, phi_x)
  $
]
#proof[
  Let $L: c2pi^k (RR^2) times c2pi (CC) times c2pi (CC) -> RR$ defined by
  $
    L(x, phi, psi) := jp(x, phi) + Re dp(psi, (I/2 + dlp_x - i eta slp_x) phi - g_x)
  $
  Then
  $
    D_x jr(x) [h] & = D_x L(x, phi_x, psi_x) [h] + D_phi L(x, phi_x, psi_x) [D_x phi_x [h]] + D_psi L(x, phi_x, psi_x) [D_x psi_x [h]]
  $
  The first term is
  $
    D_x L(x, phi_x, psi_x) [h] & = D_x jp(x, phi_x) [h] + Re dp(psi_x, D_x dlp_x [h] phi_x - i eta D_x slp_x [h] phi_x - D_x g_x [h]) \
  $
  The last two terms vanish since for any $v in c2pi$,
  $
    D_phi L(x, phi, psi_x) [v] & = D_phi jp (x, phi) [v] + Re dp(psi_x, (I/2 + dlp_x - i eta slp_x) v) \
                               & = Re dp((I/2 + dlp_x - i eta slp_x)^* psi_x + grad_phi jp (x, phi), v) = Re dp(0, v) = 0
  $
  and for any $w in c2pi$,
  $
    D_psi L(x, phi_x, psi) [w] = Re dp(w, (I/2 + dlp_x - i eta slp_x) phi_x - g_x) = Re dp(w, 0) = 0
  $
]
#remark[
  Typically $g_x := - uin compose x$, $jp (x, phi) := J(x, (dlp_x - i eta slp_x) phi)$ is used, where $uin$ is the incident wave and $J$ is the objective functional based on shape and scattered field, not density.

  In this case, $D_x g_x (t) [h] = - grad uin(x(t)) dot h(t)$, $grad_phi jp (x, phi) = (dlp_x - i eta slp_x)^* grad_u J (x, (dlp_x - i eta slp_x) phi)$, since $dp(grad_phi jp, h) = dp(grad_u J, (dlp_x - i eta slp_x) h) = dp((dlp_x - i eta slp_x)^* grad_u J, h)$
]

#remark[
  In the proof above, the step $D_phi L(x, phi_x, psi_x)[D_x phi_x[h]] = 0$ relies on the fact that the shape-induced variation $D_x phi_x[h]$ belongs to $c2pi(CC)$, the test space for which the adjoint equation holds. Here this is trivially satisfied since the boundary spaces map onto themselves smoothly; in more general Sobolev settings, verifying that variations remain valid test functions is a necessary step.
]
#algorithm[
  Assume we have implementation of $jp, D_x jp, D_phi jp, x, x', x'', h, h', h'', slp_x, dlp_x, D_x slp_x, D_x dlp_x, g_x, D_x g_x$.
  + Compute $phi_x$ by solving the boundary integral equation
  + Compute $D_phi jp$, then compute $psi_x$ by solving the adjoint equation
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

#theorem[Riesz representation for point evaluation][
  Let $x_0 in RR^2 without overline(Omega_x)$). Let $jp(x, phi) := abs(u(x))^2, u(x) := ((alpha dlp_x - i eta slp_x) phi) (x_0)$.
  Then, the Riesz representation of the Frechet derivative of $jp$ with respect to $phi$ under the sesquilinear form
  $
    dp(f, g) := integral_0^(2 pi) f(t) overline(g(t)) dd(t),
  $
  is given by
  $
    grad_phi jp(x, phi)(tau) =
    2 u(x_0) overline((alpha widetilde(D)(x_0, tau) - i eta widetilde(S)(x_0, tau))),
  $
  where
  $
    widetilde(S)(x_0, tau) := G(x_0, x(tau)) abs(x'(tau)),
    widetilde(D)(x_0, tau) := n(tau) dot grad_y G(x_0, x(tau)) abs(x'(tau))
  $
  are the kernels of $sl_x$, $dl_x$ with jacobian multiplied, evaluated at $x_0$.
]
#proof[
  Let $K(x_0, tau) := alpha widetilde(D)(x_0, tau) - i eta widetilde(S)(x_0, tau)$.
  Then
  $
    D_phi jp(x, phi)[h] & =
                          lim_(epsilon -> 0) (jp(x, phi + epsilon h) - jp(x, phi)) / epsilon \
                        & = lim_(epsilon -> 0) (abs(u + epsilon v)^2 - abs(u)^2) / epsilon \
                        & = lim_(epsilon -> 0) (abs(u)^2 + epsilon overline(u) v + epsilon u overline(v) + epsilon^2 abs(v)^2 - abs(u)^2) / epsilon \
                        & = 2 Re (overline(u) v)
  $
  where
  $
    u := u(x_0) = (alpha dlp_x - i eta slp_x) phi (x_0),
    v := (alpha dlp_x - i eta slp_x) h (x_0).
  $
  Expanding the evaluation operators gives
  $
    (alpha dlp_x - i eta slp_x) h (x_0)
    = integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau).
  $
  Therefore
  $
    D_phi jp(x, phi)[h]
    = integral_0^(2 pi)
    2 Re(overline(u(x_0)) K(x_0, tau)) h(tau) dd(tau).
  $
  By definition of the Riesz representation,
  $D_phi jp(x, phi)[h] = dp(grad_phi jp, h)$, hence
  $
    grad_phi jp(x, phi)(tau) =
    2 Re(overline(u(x_0)) K(x_0, tau)).
  $
]

#bibliography("main.bib")
