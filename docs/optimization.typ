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
  $X, Y$: $KK$-norm spaces, $forall B in B(X, Y). forall x in X. forall y in Y. dp(x, y)$ is called a bilinear form if $dp(x, y)$ is linear in $x$ and $y$. $dp$ is called non-degenerate if $forall x in X. (forall y in Y. dp(x, y) = 0) ==> x = 0$ and $forall y in Y. (forall x in X. dp(x, y) = 0) ==> y = 0$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any non-degenerate bilinear form on $c2pi(CC), c2pi(CC)$.
  Let $k >= 2$.
  Let $r in c2pi^k (RR^2), g: c2pi^k (RR^2) -> c2pi^k (CC)$.
  Let $jp: c2pi^k (RR^2) times c2pi (CC) -> RR$. Let density $phi_r$ satisfy the boundary integral equation

  $
    (I/2 + dlp_r - i eta slp_r) phi_r = g_r quad x in [0, 2pi)
  $

  Let $jr(r) := J(r, phi_r)$.
  Assume there exists $grad_phi J(r, phi_r)$ such that for any $h in c2pi (CC), D_phi jp (r, phi_r) [h] = Re dp(grad_phi jp (r, phi_r), h)$.
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
  The last two terms vanish since for any $h in c2pi$,
  $
    D_phi L(r, phi, psi_r) [h] & = D_phi jp (r, phi) [h] + Re dp(psi_r, (I/2 + dlp_r - i eta slp_r) h) \
                               & = Re dp((I/2 + dlp_r - i eta slp_r)^* psi_r + grad_phi jp (r, phi), h) = Re dp(0, h) = 0
  $
  $
    D_psi L(r, phi_r, psi) [h] = Re dp(h, (I/2 + dlp_r - i eta slp_r) phi_r - g) = Re dp(h, 0) = 0
  $
]
#remark[
  Typically $g_r := - uin compose r$, $jp (r, phi) := J(r, (dlp_r - i eta slp_r) phi)$ is used, where $uin$ is the incident wave and $J$ is the objective functional based on radius and scattered field, not density.

  In this case, $D_r g_r (x) [h] = - grad uin(r(x)) dot h(x)$, $grad_phi jp (r, phi) = (dlp_r - i eta slp_r)^* grad_u J (r, (dlp_r - i eta slp_r) phi)$, since $dp(grad_phi jp, h) = dp(grad_u J, (dlp_r - i eta slp_r) h) = dp((dlp_r - i eta slp_r)^* grad_u J, h)$
]
#algorithm[
  Assume we have implementation of $jp, D_r jp, D_phi jp, r, r', r'', h, h', h'', slp_r, dlp_r, D_r slp_r, D_r dlp_r, g_r, D_r g_r$.
  + Compute $phi_r$ by solving the boundary integral equation
  + Compute $D_phi jp$, then compute $psi_r$ by solving the adjoint equation
  + Compute $D_r jr(r) [h]$
]
#let hdj = $grad^((H)) J$
#let hdh = $h^((H))$
#definition[Hilbertian Reguralization][
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
  $
    D J (r) [hdh] & = D J (r) [-hdj(r)/norm(hdj(r))_H] = -ip(hdj(r), hdj(r) / norm(hdj(r))_H)_H \
                  & = inf_(norm(h)_H = 1) ip(hdj(r), h)_H = inf_(norm(h)_H = 1) D J (r) [h]
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

#theorem[Error estimate][
  $norm(hdr - hdk)_(h2pi^k) <=$
]

#algorithm[
  The coefficients ${c_m}_(m = 0)^(N - 1) union {d_m}_(m = 1)^(N - 1)$ of the steepest descent direction $hdk$ can be computed by

  $
    c'_m := (D_r J (r) [cos(m t)]) / (1 + alpha m^2)^k, quad d'_m := (D_r J (r) [sin(m t)]) / (1 + alpha m^2)^k
  $

  $
    S := 1/2 c'_0^2 + sum_(m = 1)^(N - 1) (1 + alpha m^2)^k (c'_m^2 + d'_m^2), quad c_m := c'_m / sqrt(S), quad d_m := d'_m / sqrt(S)
  $
]

#bibliography("main.bib")
