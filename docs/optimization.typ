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
  $X, Y$: $KK$-norm spaces, $forall O in cal(O)(X). forall F: O -> Y. D F in B(X, Y) "defined as" lim_(h -> 0) (norm(F(x + h) - F(x) - D F [h])_Y) / (norm(h)_X) = 0$
]
#definition[Bilinear form][
  $X, Y$: $KK$-norm spaces, $forall B in B(X, Y). forall x in X. forall y in Y. dp(x, y)$ is called a bilinear form if $dp(x, y)$ is linear in $x$ and $y$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any bilinear form on $c2pi(CC), c2pi(CC)$.
  Let $k >= 2$.
  Let $r in c2pi^k (RR^2), g: c2pi^k (RR^2) -> c2pi^k (CC)$.
  Let $jp: c2pi^k (RR^2) times c2pi (CC) -> RR$. Let density $phi_r$ satisfy the boundary integral equation

  $
    (I/2 + dlp_r - i eta slp_r) phi_r = g_r quad x in [0, 2pi)
  $

  Let $jr(r) := J(r, phi_r)$.
  Assume there exists $grad_phi J(r, phi_r)$ such that $forall h in c2pi (CC), D_phi jp (r, phi_r) [h] = Re dp(grad_phi J(r, phi_r), h)$.
  Then $D_r jr(r) [h]$ is given by

  $
    D_r jr(r) [h] & = D_r jp(r, phi_r) [h] + Re dp(psi_r, D_r dlp_r [h] phi_r - i eta D_r slp_r [h] phi_r - D_r g_r [h])
  $
  where $psi_r in c2pi$ satisfies the following adjoint equation:
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

  In this case, $D_r g_r (x) [h] = - grad uin(r(x)) dot h(x)$, $grad_phi jp = (dlp_r - i eta slp_r)^* grad_u J$.
]
#algorithm[
  Assume we have implementation of $jp, D_r jp, D_phi jp, r, r', r'', h, h', h'', slp_r, dlp_r, D_r slp_r, D_r dlp_r, g_r, D_r g_r$.
  + Compute $phi_r$ by solving the boundary integral equation
  + Compute $D_phi jp$, then compute $psi_r$ by solving the adjoint equation
  + Compute $D_r jr(r) [h]$
]

Periodic Sobolev space $H^3_(2 pi) subset.double C^2_(2 pi)$ is used for regularization.

#bibliography("main.bib")
