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

#definition[Multi-scatterer configuration][
  Let $M in NN$ be the number of scatterers. Let $x_j in c2pi^k (RR^2)$ for $j = 1, dots, M$, with $Gamma_j := Gamma_(x_j) := {x_j(t) | t in [0, 2pi)}$. Assume the scatterers are disjoint: $overline(Omega_(x_j)) cap overline(Omega_(x_l)) = empty$ for $j != l$.

  Let $alpha_j, eta_j in KK$ be coupling parameters for each scatterer. Define the density vector $Phi := (phi_1, dots, phi_M) in c2pi^k (CC^M)$ and the incident field vector $G := (-uin |_(Gamma_1), dots, -uin |_(Gamma_M)) in c2pi^k (CC^M)$.

  The scattered field is
  $
    u^s (x) := sum_(l = 1)^M (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l)) phi_l (x)
  $
]

#theorem[Multi-scatterer BIE][
  The combined field BIE for $M$ scatterers is the block system $A Phi = G$, where the block operator $A: c2pi^k (CC^M) -> c2pi^k (CC^M)$ has components

  $
    (A Phi)_j := (alpha_j / 2 + alpha_j dlp_(Gamma_j) - i eta_j slp_(Gamma_j)) phi_j
    + sum_(l != j) (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l)) phi_l |_(Gamma_j)
  $

  for $j = 1, dots, M$. The diagonal blocks $A_(jj) := alpha_j / 2 + alpha_j dlp_(Gamma_j) - i eta_j slp_(Gamma_j)$ are the same as the single-scatterer operator. The off-diagonal blocks $A_(jl) := (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l))|_(Gamma_j)$ for $l != j$ are smooth integral operators since $Gamma_j$ and $Gamma_l$ are disjoint.
]

#definition[Frechet derivative][
  Let $X, Y$ $KK$-norm spaces.
  Let $O subset.eq X$ be open.
  Let $F: O -> Y$, $x in X$.
  A bounded linear operator $D F (x)$ is called the Frechet derivative of $F$ at $x$ if $lim_(h -> 0) (norm(F(x + h) - F(x) - D F (x) [h])_Y) / (norm(h)_X) = 0$.
]

#definition[Sesquilinear form on product space][
  Let $X, Y$ be $KK$-norm spaces. A mapping $dp: X times Y -> KK$ is called a sesquilinear form if it is linear in the first argument and conjugate-linear in the second.

  For the product space $c2pi^k (CC^M) times c2pi^k (CC^M)$, we define the sesquilinear form
  $
    dp(Phi, Psi) := sum_(j = 1)^M integral_0^(2 pi) phi_j (t) overline(psi_j (t)) dd(t)
  $
  for $Phi = (phi_1, dots, phi_M)$ and $Psi = (psi_1, dots, psi_M)$. This form is non-degenerate: $dp(Phi, Psi) = 0$ for all $Psi$ implies $Phi = 0$, and vice versa.
]

#definition[Adjoint operator on product space][
  Let $A: c2pi^k (CC^M) -> c2pi^k (CC^M)$ be a bounded linear operator. The adjoint operator $A^*: c2pi^k (CC^M) -> c2pi^k (CC^M)$ with respect to the sesquilinear form $dp(dot, dot)$ satisfies
  $
    dp(A Phi, Psi) = dp(Phi, A^* Psi) quad forall Phi, Psi in c2pi^k (CC^M)
  $

  For the multi-scatterer BIE operator $A$, the adjoint $A^*$ is the block operator with components
  $
    (A^* Psi)_j := (alpha_j / 2 + alpha_j dlp_(Gamma_j)^* + i eta_j slp_(Gamma_j)^*) psi_j
    + sum_(l != j) (alpha_l dlp_(Gamma_l)^* + i eta_l slp_(Gamma_l)^*) psi_l |_(Gamma_j)
  $

  where $dlp_(Gamma)^*$ and $slp_(Gamma)^*$ are the adjoints of the single-scatterer operators with respect to the $L^2$ inner product on $Gamma$.
]

#theorem[Adjoint method for multiple scatterers @matsushima_2023][
  Let $dp(dot, dot)$ be the sesquilinear form on $c2pi^k (CC^M) times c2pi^k (CC^M)$ defined above. Let $k >= 2$.

  Let $X := (x_1, dots, x_M) in c2pi^k (RR^2)^M$ be the collection of scatterer boundaries. Let $J: c2pi^k (RR^2)^M times c2pi^k (CC^M) -> RR$ be Frechet differentiable. Let density $Phi_X in c2pi^k (CC^M)$ satisfy the boundary integral equation

  $
    A_X Phi_X = G_X
  $

  where $A_X$ is the multi-scatterer BIE operator and $G_X := (-uin |_(Gamma_(x_1)), dots, -uin |_(Gamma_(x_M)))$.

  Let $jr(X) := J(X, Phi_X)$. Assume there exists $grad_Phi J(X, Phi_X) in c2pi^k (CC^M)$ such that for any $H in c2pi^k (CC^M)$,
  $
    D_Phi J (X, Phi_X) [H] = Re dp(grad_Phi J (X, Phi_X), H)
  $

  Then $D_X jr(X) [H]$ for a perturbation $H = (h_1, dots, h_M) in c2pi^k (RR^2)^M$ is given by

  $
    D_X jr(X) [H] = D_X J(X, Phi_X) [H] + Re dp(Psi_X, D_X A_X [H] Phi_X - D_X G_X [H])
  $

  where $Psi_X in c2pi^k (CC^M)$ satisfies the adjoint equation
  $
    A_X^* Psi_X = - grad_Phi J (X, Phi_X)
  $
]

#proof[
  Define the Lagrangian $L: c2pi^k (RR^2)^M times c2pi^k (CC^M) times c2pi^k (CC^M) -> RR$ by
  $
    L(X, Phi, Psi) := J(X, Phi) + Re dp(Psi, A_X Phi - G_X)
  $

  Then
  $
    D_X jr(X) [H] = D_X L(X, Phi_X, Psi_X) [H] + D_Phi L(X, Phi_X, Psi_X) [D_X Phi_X [H]] + D_Psi L(X, Phi_X, Psi_X) [D_X Psi_X [H]]
  $

  The first term is
  $
    D_X L(X, Phi_X, Psi_X) [H] = D_X J(X, Phi_X) [H] + Re dp(Psi_X, D_X A_X [H] Phi_X - D_X G_X [H])
  $

  The last two terms vanish since for any $V in c2pi^k (CC^M)$,
  $
    D_Phi L(X, Phi, Psi_X) [V] = D_Phi J (X, Phi) [V] + Re dp(Psi_X, A_X V)
    = Re dp(A_X^* Psi_X + grad_Phi J (X, Phi), V) = Re dp(0, V) = 0
  $

  and for any $W in c2pi^k (CC^M)$,
  $
    D_Psi L(X, Phi_X, Psi) [W] = Re dp(W, A_X Phi_X - G_X) = Re dp(W, 0) = 0
  $
]

#remark[
  The term $D_X A_X [H] Phi_X$ represents the shape derivative of the block operator applied to the density. For the multi-scatterer case,

  $
    (D_X A_X [H] Phi)_j = sum_(l = 1)^M D_X A_(jl) [h_l] phi_l
  $

  where $D_X A_(jl) [h_l]$ is the shape derivative of the $(j, l)$ block operator with respect to perturbation of the $l$-th scatterer boundary. For $j = l$, this is the singular shape derivative of the self-interaction operator. For $j != l$, this is the smooth shape derivative of the interaction operator.
]

#remark[
  Typically $G_X := (-uin compose x_1, dots, -uin compose x_M)$ and $J (X, Phi) := J(X, u^s(Phi))$ where $u^s(Phi)$ is the scattered field and $J$ is the objective functional based on shape and scattered field, not density.

  In this case, $D_X G_X [H] = (-grad uin(x_1) dot h_1, dots, -grad uin(x_M) dot h_M)$, and the gradient $grad_Phi J$ is computed via the chain rule through the scattered field evaluation.
]

#algorithm[Multi-scatterer optimization][
  Assume we have implementation of $J, D_X J, D_Phi J, X, X', X'', H, H', H'', A_X, A_X^*, D_X A_X, G_X, D_X G_X$.
  + Solve the forward BIE: $A_X Phi_X = G_X$
  + Compute $D_Phi J$, then solve the adjoint equation: $A_X^* Psi_X = - grad_Phi J (X, Phi_X)$
  + Compute the shape derivative: $D_X jr(X) [H] = D_X J(X, Phi_X) [H] + Re dp(Psi_X, D_X A_X [H] Phi_X - D_X G_X [H])$
  + Compute the Riesz representative $hdj(X)$ of $D_X jr(X)$ to obtain the gradient for each scatterer
  + Update the shapes: $(x_j)_(n + 1) = (x_j)_n + lambda (hdh_j)$ where $hdh_j := - hdj_j(X) / norm(hdj_j(X))_H$
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

#definition[Hilbert space for multiple scatterers][
  For $M$ scatterers, we use the product Hilbert space $H^M := h2pi^k (RR)^M$ with inner product
  $
    ip(Phi, Psi)_(H^M) := sum_(j = 1)^M ip(phi_j, psi_j)_h2pi^k
  $
  for $Phi = (phi_1, dots, phi_M)$ and $Psi = (psi_1, dots, psi_M)$. The norm is
  $
    norm(Phi)_(H^M)^2 = sum_(j = 1)^M norm(phi_j)_(h2pi^k)^2
  $
]

#let hdr = $h^((R_N))$
#let hdk = $h^((h2pi^k))$
#definition[
  Let $R_N := {a_0 / 2 + sum_(m = 1)^(N - 1) (a_m cos(m t) + b_m sin(m t)) | a_m, b_m in RR} subset.neq h2pi^k (RR)$.
  $R_N$ is continuously embedded in $h2pi^k (RR)$.
]

#theorem[
  The coefficients ${c_(j,m)}_(m = 0)^(N - 1) union {d_(j,m)}_(m = 1)^(N - 1)$ of the finite-dimensional steepest descent direction $hdr$ for the $j$-th scatterer can be computed by

  $
    c'_(j,m) := (D_(x_j) J (X) [cos(m t)]) / (1 + alpha m^2)^k, quad d'_(j,m) := (D_(x_j) J (X) [sin(m t)]) / (1 + alpha m^2)^k
  $

  $
    S_j := 1/2 c'_(j,0)^2 + sum_(m = 1)^(N - 1) (1 + alpha m^2)^k (c'_(j,m)^2 + d'_(j,m)^2), quad c_(j,m) := c'_(j,m) / sqrt(S_j), quad d_(j,m) := d'_(j,m) / sqrt(S_j)
  $

  where $c'_(j,m), d'_(j,m)$ are the Fourier coefficients of the unnormalized Riesz representation $hdj_j(X)$ for the $j$-th scatterer.
]

#theorem[Error estimate][
  Let $g_j := hdj_j(X)$ be the unnormalized Riesz representation in $h2pi^k(RR)$ and $g_(j,N)$ its truncation in $R_N$. If $g_j in h2pi^(k + s)(RR)$ for some $s > 0$, then
  $
    norm(g_j - g_(j,N))_(h2pi^k) <= (1 + alpha N^2)^(-s/2) norm(g_j)_(h2pi^(k + s))
  $
]

#proof[
  Let $c'_(j,m), d'_(j,m)$ be the Fourier coefficients of $g_j$ as defined above.
  The squared norm of the truncation error is the tail of the series:
  $
    norm(g_j - g_(j,N))_(h2pi^k)^2 & = sum_(m = N)^infinity (1 + alpha m^2)^k ((c'_(j,m))^2 + (d'_(j,m))^2) \
                                   & = sum_(m = N)^infinity (1 + alpha m^2)^(-s) (1 + alpha m^2)^(k + s) ((c'_(j,m))^2 + (d'_(j,m))^2) \
                                   & <= (1 + alpha N^2)^(-s) sum_(m = N)^infinity (1 + alpha m^2)^(k + s) ((c'_(j,m))^2 + (d'_(j,m))^2) \
                                   & <= (1 + alpha N^2)^(-s) norm(g_j)_(h2pi^(k + s))^2
  $
  Taking square roots yields the claimed bound.
]

#let ju = $tilde(J)$
#theorem[Riesz representation for point evaluation with multiple scatterers][
  Let $x_0 in RR^2 without bigcup_(j = 1)^M overline(Omega_(x_j))$.
  Let $ju in C^1 (RR^2, RR)$.
  Let $J(X, Phi) := ju(Re u^s_Phi (x_0), Im u^s_Phi (x_0))$, where
  $
    u^s_Phi (x) := sum_(l = 1)^M (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l)) phi_l (x)
  $

  Then, the Riesz representation of the Frechet derivative of $J$ with respect to $Phi$ under the sesquilinear form
  $
    dp(Phi, Psi) := sum_(j = 1)^M integral_0^(2 pi) phi_j(t) overline(psi_j(t)) dd(t)
  $

  is given by
  $
    (grad_Phi J(X, Phi))_j (tau) =
    (pdv(ju, x_1) + i pdv(ju, x_2)) (u^s_Phi (x_0)) overline(K_j(x_0, tau))
  $

  where $pdv(ju, x_1), pdv(ju, x_2)$ are the partial derivatives of $ju$ with respect to its first and second arguments, and
  $
    K_j(x_0, tau) := alpha_j tilde(D)_j(x_0, tau) - i eta_j tilde(S)_j(x_0, tau)
  $

  with
  $
    tilde(S)_j(x_0, tau) := G(x_0, x_j(tau)) abs(x_j'(tau)),
    tilde(D)_j(x_0, tau) := n_j(tau) dot grad_y G(x_0, x_j(tau)) abs(x_j'(tau))
  $

  the kernels of $sl_(Gamma_j)$, $dl_(Gamma_j)$ with jacobian multiplied, evaluated at $x_0$.
]

#proof[
  Let $u := u^s_Phi (x_0)$ and $K_j(x_0, tau) := alpha_j tilde(D)_j(x_0, tau) - i eta_j tilde(S)_j(x_0, tau)$.
  Since
  $
    D_Phi J(X, Phi)[H] = D ju(u) [sum_(l = 1)^M (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l)) h_l (x_0)]
  $

  and expanding the evaluation operators gives
  $
    sum_(l = 1)^M (alpha_l dlp_(Gamma_l) - i eta_l slp_(Gamma_l)) h_l (x_0)
    = sum_(l = 1)^M integral_0^(2 pi) K_l(x_0, tau) h_l(tau) dd(tau)
  $

  using that $ju$ is real-valued, hence $D ju$ is real-linear,
  $
    D_Phi J(X, Phi)[H] & = pdv(ju, x_1)(u) Re sum_(l = 1)^M integral_0^(2 pi) K_l(x_0, tau) h_l(tau) dd(tau) \
                       & quad + pdv(ju, x_2)(u) Im sum_(l = 1)^M integral_0^(2 pi) K_l(x_0, tau) h_l(tau) dd(tau) \
                       & = Re sum_(l = 1)^M integral_0^(2 pi) (pdv(ju, x_1) - i pdv(ju, x_2))(u) K_l(x_0, tau) h_l(tau) dd(tau) \
                       & = Re sum_(l = 1)^M integral_0^(2 pi) (pdv(ju, x_1) + i pdv(ju, x_2))(u) overline(K_l(x_0, tau)) overline(h_l(tau)) dd(tau)
  $

  where the last equality uses $Re(z) = Re(overline(z))$.
  Comparing with $Re dp(grad_Phi J, H) = Re sum_(j = 1)^M integral_0^(2 pi) (grad_Phi J)_j (t) overline(h_j(t)) dd(t)$ gives the claimed representation.
]

#bibliography("main.bib")
