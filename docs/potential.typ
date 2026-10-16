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
#let slpo = $caron(slp)$
#let dlpo = $caron(dlp)$
#let dlpao = $caron(dlpa)$
#let tlpo = $caron(tlp)$
#let slo = $caron(sl)$
#let dlo = $caron(dl)$
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
  Let
  $
    G(xi, chi) := i/4 hk1_0 (k abs(xi - chi)).
  $
  Let $Omega$ be a bounded open set which has connected $C^2$ boundary $Gamma$. Let
  $
     slpo: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma G(xi, chi) phi(chi) dd(chi) \
     dlpo_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(chi) dot grad_chi G(xi, chi) phi(chi) dd(chi) \
    dlpao_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral_Gamma n(xi) dot grad_xi G(xi, chi) phi(chi) dd(chi) \
     tlpo_Gamma: C(Gamma) -> C(Gamma), phi & |-> integral.dash_Gamma sum_(i,j) n_i (xi) n_j (chi) pdv(G, xi_i, chi_j)(xi, chi) phi(chi) dd(chi) \
     slo_Gamma : C(Gamma) -> C(outomega), phi &|-> integral_Gamma G(xi, chi) phi(chi) dd(chi) \
     dlo_Gamma : C(Gamma) -> C(outomega), phi &|-> integral_Gamma n(chi) dot grad_chi G(xi, chi) phi(chi) dd(chi)
  $
]


#theorem[
  Let $(Omega_a)_(a in I)$ be bounded open sets satisfying $forall a, b in I. overline(Omega_a) inter overline(Omega_b) = emptyset$ with connected $C^2$ boundaries $(Gamma_a)_(a in I)$.
  Let $Omega := union_(a in I) Omega_a$, $Gamma := union_(a in I) Gamma_a$.
  Let $k > 0$, let $eta > 0$, let $u in C^1 (RR^2 without Omega)$ satisfy
  $
  cases(
    (laplacian + k^2) u = 0 "in" outomega,
    u = -uin "on" Gamma,
    "SRC" "on" SS^1
  ).
  $
  Then there exists $phi in C(Gamma)$ such that
  $
  cases(
    (I/2 + dlpo_Gamma - i eta slpo_Gamma) phi(xi) = f(xi) := - uin(xi) &quad (xi in Gamma),
    u(xi) = (dlo_Gamma - i eta slo_Gamma) phi(xi) &quad(xi in RR^d without overline(Omega))
  )
  $
] <bie-original>
#definition[
  Let $c2pi^k (KK) := C^k (RR \/ 2 pi, KK)$.
  Let $B := {(x_1, x_2) in c2pi^2 times c2pi^2 | forall t in [0, 2 pi). abs(x'(t)) != 0 and "counterclockwise" and "does not intersect itself"}$.
  Let $Gamma_dot: B -> 2^RR^2, x |-> {x(t) | t in [0, 2 pi)}$.
]
#definition[
  Let
  $
    G(x, y) := i/4 hk1_0 (k abs(x - y)),
  $
  $
    slp_x&: c2pi -> c2pi, psi |-> slp_Gamma_x (psi compose x) compose x^(-1), \
    dlp_x&: c2pi -> c2pi, psi |-> dlp_Gamma_x (psi compose x) compose x^(-1), \
    dlpa_x&: c2pi -> c2pi, psi |-> dlpa_Gamma_x (psi compose x) compose x^(-1), \
    tlp_x&: c2pi -> c2pi, psi |-> tlp_Gamma_x (psi compose x) compose x^(-1), \
    sl_x&: c2pi -> c2pi, psi |-> slo_Gamma_x (psi compose x) compose x^(-1), \
    dl_x&: c2pi -> c2pi, psi |-> dlo_Gamma_x (psi compose x) compose x^(-1).
  $
]
#theorem[
  Let $g_x (t) := f(x(t)) = -uin(x(t))$.
  Let $phi$ be a solution of @bie-original.
  Let $I$ be a finite index set.
  Let $(x_a)_(a in I) subset B$.
  Let $(psi_a)_(a in I) := (psi compose x_a)$. Then
  $
  cases(
  (I/2 + dlp_x_a - i eta slp_x_a + sum_(b in I, b != a) dl_x_a - i eta sl_x_a) psi_a = g_x_a quad (a in I),
  sum_(a in I) (dl_x_a - i eta sl_x_a)psi_a in outomega
  )
  $
] <bie>
#definition[Counterclockwise outward vectors][
  Let $x in B$. Let
  $
    n_x (t) := (n_x^*(t))/ abs(x'(t)), quad n_x^*(t) := (x'_2(t), -x'_1(t)).
  $
]
#theorem[@zhang_superconvergence_2010][
  $
                                      grad_chi G(xi, chi) & = (i k)/4 (hk1_1 (k abs(xi - chi)))/(abs(xi - chi)) (xi - chi) \
    sum_(i,j) n_i (xi) n_j (chi) pdv(G, xi_i, chi_j)(xi, chi) & = (i k)/4 hk1_0 (k abs(xi - chi)) (k n(xi) dot (xi - chi) n(chi) dot (xi - chi)) / (abs(xi - chi)^2) \
                                                     & + (i k)/4 hk1_1 (k abs(xi - chi)) ((n(xi) dot n(chi))/(abs(xi - chi)) - 2 (n(xi) dot (xi - chi) n(chi) dot (xi - chi))/(abs(xi - chi)^3))
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
#let slk = $op("sl")$
#let dlk = $op("dl")$
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
  Let $X, Y$ be $KK$-norm spaces.
  $dp(x, y)$ is called a sesquilinear form if
  it is linear in $x$ and conjugate-linear in $y$.
  $dp(dot, dot)$ is called non-degenerate if $forall x in X. (forall y in Y. dp(x, y) = 0) ==> x = 0$ and $forall y in Y. (forall x in X. dp(x, y) = 0) ==> y = 0$.
]
#theorem[Adjoint method @matsushima_2023][
  Let $dp(dot, dot)$ any non-degenerate sesquilinear form on $c2pi(CC)^2$.
  A superscript star denotes the adjoint with respect to $dp(dot, dot)$.
  Let $k >= 2$.
  Let $I$, $x := (x_a)_(a in I)$, $psi_x := (psi_(x_a))_(a in I)$ be those in @bie.
  Let $J: B^I times c2pi (CC)^I -> RR$ be Frechet differentiable.
  Let $jr(x) := J(x, psi_x)$.
  Assume that for every $a in I$ there exists $grad_(psi_a) J(x, psi_x) in c2pi(CC)$ such that for any $v_a in c2pi(CC)$,
  $
    D_(psi_a) J(x, psi_x)[v_a]
      = Re dp(grad_(psi_a) J(x, psi_x), v_a).
  $
  Then, for every $a in I$,
  $
    D_(x_a) jr(x)[h_a]
      & = D_(x_a) J(x, psi_x)[h_a] \
      & + Re dp(lambda_a,
          (D_(x_a) dlp_(x_a)[h_a] - i eta D_(x_a) slp_(x_a)[h_a]) psi_(x_a) - D_(x_a) g_(x_a)[h_a]) \
      &  + Re dp(lambda_a, sum_(b in I, b != a) D_(x_a) [((dl_(x_b) - i eta sl_(x_b)) psi_(x_b)) compose x_a][h_a]) \
      & + Re sum_(b in I, b != a) dp(lambda_b,
          ((D_(x_a) dl_(x_a)[h_a] - i eta D_(x_a) sl_(x_a)[h_a]) psi_(x_a)) compose x_b).
  $
  Here $(lambda_a)_(a in I) in c2pi(CC)^I$ satisfies the adjoint equations,
  $
    (I/2 + dlp_(x_a) - i eta slp_(x_a))^* lambda_a
      + sum_(b in I, b != a) [((dl_(x_a) - i eta sl_(x_a)) dot) compose x_b]^* lambda_b
      = - grad_(psi_a) J(x, psi_x) quad (a in I).
  $
]
#proof[
  Define
  $
    L((x_a)_(a in I), (psi_a)_(a in I), (lambda_a)_(a in I))
      := J((x_a)_(a in I), (psi_a)_(a in I)) \
      + Re sum_(a in I) dp(lambda_a,
          (I/2 + dlp_(x_a) - i eta slp_(x_a)) psi_a
          + sum_(b in I, b != a) ((dl_(x_b) - i eta sl_(x_b)) psi_b) compose x_a
          - g_(x_a)).
  $
  Differentiation with respect to $x_a$ gives the displayed formula for every $a in I$ and $v_a in c2pi(CC)$,
  $
    D_(psi_a) L[v_a]
      &= Re dp(
          grad_(psi_a) J(x, psi_x)
          + (I/2 + dlp_(x_a) - i eta slp_(x_a))^* lambda_a, v_a) \
          &quad + Re dp(sum_(b in I, b != a) [((dl_(x_a) - i eta sl_(x_a)) dot) compose x_b]^* lambda_b,
          v_a) \
          &= 0
  $
  For every $a in I$ and $w_a in c2pi(CC)$, differentiation with respect to $lambda_a$ gives
  $
    D_(lambda_a) L[w_a]
      = Re dp(w_a,
          (I/2 + dlp_(x_a) - i eta slp_(x_a)) psi_(x_a)
          + sum_(b in I, b != a) ((dl_(x_b) - i eta sl_(x_b)) psi_(x_b)) compose x_a
          - g_(x_a)) = 0.
  $
]
#remark[
  Typically $g_(x_a) := - uin compose x_a$, and the scattered field is
  $
    u := sum_(a in I) (dl_(x_a) - i eta sl_(x_a)) psi_(x_a).
  $
  In this case, $D_(x_a) g_(x_a)(t)[h_a] = - grad uin(x_a(t)) dot h_a(t)$. The density-gradient component with index $a$ is obtained by applying the adjoint of $(dl_(x_a) - i eta sl_(x_a))$ to $grad_u J$.
]
#algorithm[
  Assume the boundary operators and their shape derivatives are implemented for all pairs of scatterers.
  + Compute $(psi_(x_a))_(a in I)$ by solving the coupled boundary integral equations
  + Compute $(grad_(psi_a) J)_(a in I)$, then compute $(lambda_a)_(a in I)$ by solving the coupled adjoint equations
  + Compute the Riesz representative of $D_(x_a) jr(x)$ for every $a in I$
  + Update every $x_a$ using the resulting regularized descent direction
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
  Let $a_m (f) := 1/pi integral_0^(2 pi) f(t) cos(m t) dd(t)$, $b_m (f) := 1/pi integral_0^(2 pi) f(t) sin(m t) dd(t)$.
  Let $ip(f, g)_h2pi^k := 1/2 a_0 (f) a_0 (g) + sum_(m = 1)^infinity (1 + alpha m^2)^k (a_m (f) a_m (g) + b_m (f) b_m (g))$.
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
  Let $J(x, psi) := ju(Re u_psi (x), Im u_psi (x))$.

  Then, the Riesz representation of the Frechet derivative of $J$ with respect to $psi$ under the sesquilinear form
  $
    dp(f, g) := integral_0^(2 pi) f(t) overline(g(t)) dd(t),
  $
  is given by
  $
    grad_psi J(x, psi)(tau) =
    (pdv(ju, x_1) + i pdv(ju, x_2)) (u_psi (x_0)) overline(K(x_0, tau)),
  $
  where $pdv(ju, x_1), pdv(ju, x_2)$ are the partial derivatives of $ju$ with respect to its first and second arguments, and
  $
    K(x_0, tau) := dlk(x_0, tau) - i eta slk(x_0, tau).
  $
  Especially, if $c in CC$ and $ju(u) = abs(u - c)^2 = (Re u - Re c)^2 + (Im u - Im c)^2$, then $(pdv(ju, x_1) + i pdv(ju, x_2))(u) = 2 (u - c)$.
]
#proof[
  Let $u := u_psi (x_0)$.
  Since
  $
    D_psi J(x, psi)[h] = D ju(u) [(dlp_x - i eta slp_x) h (x_0)]
  $
  and expanding the evaluation operators gives
  $
    (dlp_x - i eta slp_x) h (x_0)
    = integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau),
  $
  using that $ju$ is real-valued, hence $D ju$ is real-linear,
  $
    D_psi J(x, psi)[h] & = pdv(ju, x_1)(u) Re integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau) + pdv(ju, x_2)(u) Im integral_0^(2 pi) K(x_0, tau) h(tau) dd(tau) \
                       & = Re integral_0^(2 pi) (pdv(ju, x_1) - i pdv(ju, x_2))(u) K(x_0, tau) h(tau) dd(tau) \
                       & = Re integral_0^(2 pi) (pdv(ju, x_1) + i pdv(ju, x_2))(u) overline(K(x_0, tau)) overline(h(tau)) dd(tau),
  $
  where the last equality uses $Re(z) = Re(overline(z))$.
  Comparing with $Re dp(grad_psi J, h)$ gives the claimed representation.
]

#bibliography("main.bib")
