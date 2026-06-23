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
== Boundary integral equation @colton_inverse_2019 @matsushima_2023

#let c2pi = $C_(2 pi)$
#definition[
  $
    slp_r phi (x) := integral_Gamma G(x, y) phi(y) dd(s(y)), quad dlp_r phi (x) := integral_Gamma pdv(G(x, y), n(y)) phi(y) dd(s(y)), quad G(x, y) := i/4 hk1(k abs(x - y))
  $
]

#theorem[Adjoint method @matsushima_2023][
  Let $k >= 2$.
  Let $r, g in C_(2 pi)^k$.
  Let $jp: C_(2 pi)^k times c2pi -> RR$. Let $phi_r$ satisfy the boundary integral equation

  $
    (I/2 + dlp_r - i eta slp_r) phi_r = g (:= - uin compose r) quad x in [0, 2pi)
  $

  Let $jr(r) := J(r, phi_r)$, then

  $
    D_r jr(r) [h] & = D_r jp(r, phi_r) [h] + dp(psi_r, D_r dlp_r [h] phi_r - i eta D_r slp_r [h] phi_r - D_r g[h])_(c2pi, c2pi)
  $
  where $psi_r in c2pi$ satisfies the following adjoint equation:
  $
    (I/2 + dlp_r - i eta slp_r)^* psi_r = - (D_phi jp) (r, phi_r)
  $
]
#proof[
  Let $L: C_(2 pi)^k times C_(2 pi) times C_(2 pi) -> RR$ defined by
  $
    L(r, phi, psi) := jp(r, phi) + dp(psi, (I/2 + dlp_r - i eta slp_r) phi - g)_(c2pi,c2pi)
  $
  Then
  $
    D_r jr(r) [h] & = D_r L(r, phi_r, psi_r) [h] + D_phi L(r, phi_r, psi_r) [D_r phi_r [h]] + D_psi L(r, phi_r, psi_r) [D_r psi_r [h]]
  $
  The first term is
  $
    D_r L(r, phi_r, psi_r) [h] & = D_r jp(r, phi_r) [h] + dp(psi_r, D_r dlp_r [h] phi_r - i eta D_r slp_r [h] phi_r - D_r g[h])_(c2pi, c2pi) \
  $
  The last two terms vanish since
  $
    D_phi L(r, phi, psi_r) [h] & = D_phi jp (r, phi) [h] + dp(psi_r, (I/2 + dlp_r - i eta slp_r) h) \
                               & = dp((I/2 + dlp_r - i eta slp_r)^* psi_r + D_phi jp (r, phi), h)_(c2pi, c2pi) = dp(0, h)_(c2pi, c2pi) = 0
  $
  $
    D_psi L(r, phi_r, psi) [h] = dp(h, (I/2 + dlp_r - i eta slp_r) phi_r - g)_(c2pi, c2pi) = dp(h, 0)_(c2pi, c2pi) = 0
  $
]

Periodic Sobolev space $H^3_(2 pi) subset.double C^2_(2 pi)$ is used for regularization.

#bibliography("main.bib")
