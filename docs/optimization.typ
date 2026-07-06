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
  Let $c2pi^k (KK) := C^k (RR \/ 2 pi, KK)$, $r in c2pi^k (RR^2)$, $Gamma_r := {r(t) | t in [0, 2pi)}$.
  $
    slp_r phi (x) := integral_Gamma_r G(x, y) phi(y) dd(s(y)), quad dlp_r phi (x) := integral_Gamma_r pdv(G(x, y), n(y)) phi(y) dd(s(y)), quad G(x, y) := i/4 hk1_0 (k abs(x - y))
  $
]
#definition[Frechet derivative][
  Let $X, Y$ $KK$-norm spaces.
  Let $O subset.eq X$ be open.
  Let $F: O -> Y$, $x in X$.
  A bounded linear operator $D F (x)$ is called the Frechet derivative of $F$ at $x$ if $lim_(h -> 0) (norm(F(x + h) - F(x) - D F (x) [h])_Y) / (norm(h)_X) = 0$.
]
#definition[Bilinear form][
  $X, Y$: $KK$-norm spaces, $forall B in B(X, Y). dp(x, y)$ is called a bilinear form if $dp(x, y)$ is linear in $x$ and $y$. $dp$ is called non-degenerate if $forall x in X. (forall y in Y. dp(x, y) = 0) ==> x = 0$ and $forall y in Y. (forall x in X. dp(x, y) = 0) ==> y = 0$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any non-degenerate bilinear form on $c2pi(CC), c2pi(CC)$. Let $A^*$ adjoint operator of $A$ with respect to $dp(dot, dot)$.
  Let $k >= 2$.
  Let $r in c2pi^k (RR^2), g: c2pi^k (RR^2) -> c2pi^k (CC)$.
  Let $jp: c2pi^k (RR^2) times c2pi (CC) -> RR$ Frechet differentiable. Let density $phi_r in c2pi^k$ satisfy the boundary integral equation

  $
    (I/2 + dlp_r - i eta slp_r) phi_r = g_r quad x in [0, 2pi)
  $

  Let $jr(r) := J(r, phi_r)$.
  Assume there exists $grad_phi J(r, phi_r) in c2pi (RR)$ such that for any $h in c2pi (RR), D_phi jp (r, phi_r) [h] = dp(grad_phi jp (r, phi_r), h)$.
  Then $D_r jr(r) [h]$ is given by

  $
    D_r jr(r) [h] & = D_r jp(r, phi_r) [h] + Re dp(psi_r, D_r dlp_r [h] phi_r - i eta D_r slp_r [h] phi_r - D_r g_r [h])
  $
  where $psi_r in c2pi (CC)$ satisfies the following adjoint equation:
  $
    (I/2 + dlp_r - i eta slp_r)^* psi_r = - grad_phi jp (r, phi_r)
  $
]
#proof[
  Let $L: c2pi^k (RR^2) times c2pi (CC) times c2pi (CC) -> RR$ defined by
  $
    L(r, phi, psi) := jp(r, phi) + Re dp(psi, (I/2 + dlp_r - i eta slp_r) phi - g_r)
  $
  Then
  $
    D_r jr(r) [h] & = D_r L(r, phi_r, psi_r) [h] + D_phi L(r, phi_r, psi_r) [D_r phi_r [h]] + D_psi L(r, phi_r, psi_r) [D_r psi_r [h]]
  $
  The first term is
  $
    D_r L(r, phi_r, psi_r) [h] & = D_r jp(r, phi_r) [h] + Re dp(psi_r, D_r dlp_r [h] phi_r - i eta D_r slp_r [h] phi_r - D_r g_r [h]) \
  $
  The last two terms vanish since for any $v in c2pi$,
  $
    D_phi L(r, phi, psi_r) [v] & = D_phi jp (r, phi) [v] + Re dp(psi_r, (I/2 + dlp_r - i eta slp_r) v) \
                               & = Re dp((I/2 + dlp_r - i eta slp_r)^* psi_r + grad_phi jp (r, phi), v) = Re dp(0, v) = 0
  $
  and for any $w in c2pi$,
  $
    D_psi L(r, phi_r, psi) [w] = Re dp(w, (I/2 + dlp_r - i eta slp_r) phi_r - g) = Re dp(w, 0) = 0
  $
]
#remark[
  Typically $g_r := - uin compose r$, $jp (r, phi) := J(r, (dlp_r - i eta slp_r) phi)$ is used, where $uin$ is the incident wave and $J$ is the objective functional based on radius and scattered field, not density.

  In this case, $D_r g_r (x) [h] = - grad uin(r(x)) dot h(x)$, $grad_phi jp (r, phi) = (dlp_r - i eta slp_r)^* grad_u J (r, (dlp_r - i eta slp_r) phi)$, since $dp(grad_phi jp, h) = dp(grad_u J, (dlp_r - i eta slp_r) h) = dp((dlp_r - i eta slp_r)^* grad_u J, h)$
]

#remark[
  In the proof above, the step $D_phi L(r, phi_r, psi_r)[D_r phi_r[h]] = 0$ relies on the fact that the shape-induced variation $D_r phi_r[h]$ belongs to $c2pi(CC)$, the test space for which the adjoint equation holds. Here this is trivially satisfied since the boundary spaces map onto themselves smoothly; in more general Sobolev settings, verifying that variations remain valid test functions is a necessary step.
]
#algorithm[
  Assume we have implementation of $jp, D_r jp, D_phi jp, r, r', r'', h, h', h'', slp_r, dlp_r, D_r slp_r, D_r dlp_r, g_r, D_r g_r$.
  + Compute $phi_r$ by solving the boundary integral equation
  + Compute $D_phi jp$, then compute $psi_r$ by solving the adjoint equation
  + Compute the Riesz representative $hdj(r)$ of $D_r jr(r)$ to obtain the gradient (e.g. via the spectral coefficients $c'_m, d'_m$)
  + Update the shape: $r_(n + 1) = r_n + lambda hdh$ where $hdh := - hdj(r) / norm(hdj(r))_H$
]
#definition[Hilbertian Regularization][
  Let $X$ be a norm space.
  Let $J: X -> RR$ be Frechet differentiable at $x in X$.
  Let $H subset.eq X$ be a Hilbert space continuously embedded in $X$.
  Since $H$ is a Hilbert space, there exists a Riesz representation $hdj: X -> H$ such that for any $r in X, h in H$,

  $
    ip(hdj(r), h)_H = D J (r) [h] quad forall h in H
  $
  The regularized steepest descent direction $hdh$ is defined as
  $
    hdh := - hdj(r)/norm(hdj(r))_H
  $
]
#theorem[
  The regularized steepest descent direction $hdh$ is the steepest descent direction with respect to $norm(dot)_H$, i.e. $hdh = arginf_(norm(h)_H = 1) D J (r) [h]$.
]
#proof[
  By the Cauchy–Schwarz inequality,
  $
    D J (r) [h] = ip(hdj(r), h)_H >= -norm(hdj(r))_H norm(h)_H = -norm(hdj(r))_H
  $
  for any $norm(h)_H = 1$, with equality if and only if $h = -hdj(r) / norm(hdj(r))_H$.
  Hence
  $
    D J (r) [hdh] = D J (r) [-hdj(r)/norm(hdj(r))_H] = -norm(hdj(r))_H = inf_(norm(h)_H = 1) D J (r) [h]
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
    c'_m := (D_r J (r) [cos(m t)]) / (1 + alpha m^2)^k, quad d'_m := (D_r J (r) [sin(m t)]) / (1 + alpha m^2)^k
  $

  $
    S := 1/2 c'_0^2 + sum_(m = 1)^(N - 1) (1 + alpha m^2)^k (c'_m^2 + d'_m^2), quad c_m := c'_m / sqrt(S), quad d_m := d'_m / sqrt(S)
  $
  where $c'_m, d'_m$ are the Fourier coefficients of the unnormalized Riesz representation $hdj(r)$.
]

#theorem[Error estimate][
  Let $g := hdj(r)$ be the unnormalized Riesz representation in $h2pi^k(RR)$ and $g_N$ its truncation in $R_N$. If $g in h2pi^(k + s)(RR)$ for some $s > 0$, then
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

#remark[Riesz representation of the Frechet derivative with respect to $phi$ for point evaluation][
  Consider the objective
  $
    jp(r, phi) := |(alpha dlp_r - i eta slp_r) phi (x_0)|^2,
  $
  where $x_0$ is a fixed exterior point.
  Let the bilinear form be
  $
    dp(f, g) := integral_0^(2 pi) f(t) g(t) dd(t).
  $
  The Frechet derivative of $jp$ with respect to $phi$ is
  $
    D_phi jp(r, phi)[h] =
    2 Re(overline((alpha dlp_r - i eta slp_r) phi (x_0)) (alpha dlp_r - i eta slp_r) h (x_0)).
  $
  Writing the combined evaluation kernel
  $
    K(x_0, tau) :=
    alpha tilde(D)(x_0, tau) - i eta tilde(S)(x_0, tau)
  $
  where $tilde(D), tilde(S)$ are the parametrised evaluation kernels of
  the double- and single-layer potentials at $x_0$, the derivative becomes
  $
    D_phi jp(r, phi)[h] =
    integral_0^(2 pi)
    2 Re(overline(u(x_0)) K(x_0, tau)) h(tau) dd(tau).
  $
  Hence the Riesz representation is
  $
    grad_phi jp(r, phi)(tau) =
    2 Re(overline(u(x_0)) K(x_0, tau)).
  $
]

#bibliography("main.bib")
