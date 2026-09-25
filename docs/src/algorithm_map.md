# Algorithm map

This page lists, for every step of the algorithm of [AKN, §7.1, pp. 53–57], the functions of
Syntomic.jl that implement it. [AKN] is B. Antieau, A. Krause, T. Nikolaus, *On the K-theory of
Z/p^n*, arXiv:2405.04329; page numbers are those of the paper.

The driver [`Syntomic.syntomic_wmatrices`](@ref) and [`Syntomic.syntomic_square`](@ref) call the
steps in order; [`syntomic_complex`](@ref) and [`syntomic_cohomology`](@ref) wrap them.

| Step | Content | Functions | File | Reference |
|:-----|:--------|:----------|:-----|:----------|
| Input | the ring `O_K/ϖ^n`, normalization `E(0) = p` | [`finite_chain_ring`](@ref), [`Syntomic.normalize_eisenstein`](@ref), [`Syntomic.EisensteinData`](@ref) | `input.jl`, `eisenstein.jl` | [AKN, §7.1, Input, p. 53] |
| 1 | working precision | [`Syntomic.target_precision`](@ref), [`Syntomic.apriori_working_precision`](@ref), [`Syntomic.exponent_working_precision`](@ref), [`Syntomic.working_precision`](@ref) | `precision.jl`, `pipeline.jl` | [AKN, §7.1, Step 1, p. 54; §7.2, pp. 57–60] |
| 2 | linear bases; `W(F_q) = Z_p[x]/(h̃)` | [`SyntomicParams`](@ref), [`Syntomic.f_basis_exponents`](@ref), [`Syntomic.g_basis_exponents`](@ref), [`Syntomic.GaloisRing`](@ref), [`Syntomic.zp_expand`](@ref) | `params.jl`, `arith/galois_ring.jl`, `linalg/wmatrix.jl` | [AKN, §7.1, Step 2, pp. 54–55; Prop. 3.34, p. 24] |
| 3 | δ-structure on `W(F_q)[[z_0]]` | [`Syntomic.ws_frobenius`](@ref), [`Syntomic.ws_delta`](@ref) | `arith/wseries.jl` | [AKN, §2.3, p. 11; §3.1, p. 14] |
| 3 | the units `λ_u` | [`Syntomic.LambdaData`](@ref), [`Syntomic.lambda_zero`](@ref), [`Syntomic.lambda_denominator`](@ref), [`Syntomic.lambda_next`](@ref) | `step3/lambdas.jl` | [AKN, Lemma 3.24, p. 21] |
| 3 | envelope of `R = O_K/ϖ^n` over `W(F_q)[[z_0]]` | [`Syntomic.FEnvelope`](@ref), [`Syntomic.f_relation_coefficient`](@ref), [`Syntomic.f_monomial_normal_form`](@ref), [`Syntomic.f_mul`](@ref) | `step3/envelope_f.jl` | [AKN, Lemma 3.24, 3.26, Remark 3.29, pp. 21–22] |
| 3 | envelope of `O_K` over `W(F_q)[[z_0, z_1]]` | [`Syntomic.GEnvelope`](@ref), [`Syntomic.g_Rprime_zero`](@ref), [`Syntomic.g_Rprime_next`](@ref), [`Syntomic.g_relation`](@ref), [`Syntomic.g_frobenius_generator`](@ref) | `step3/envelope_g.jl` | [AKN, Lemma 3.24, 3.26, Construction 3.28, pp. 21–22] |
| reduce | normal forms in the bases of Step 2 | [`Syntomic.f_monomial_normal_form`](@ref), [`Syntomic.g_monomial_normal_form`](@ref), [`Syntomic.g_mul`](@ref), [`Syntomic.g_frobenius`](@ref), [`Syntomic.g_delta`](@ref) | `step3/*.jl` | [AKN, §7.1, reduce subalgorithm, p. 55] |
| 4 | the unit `w(v)` | [`Syntomic.divided_difference_E`](@ref), [`Syntomic.phi_one_g0`](@ref), [`Syntomic.bk_unit_u`](@ref), [`Syntomic.bk_unit_v`](@ref), [`Syntomic.bk_unit_power`](@ref) | `step4_orientation.jl` | [AKN, §7.1, Step 4, pp. 55–56; Lemma 4.15, p. 29] |
| 5 | `∇` for `O_K` | [`Syntomic.nabla_theta_table`](@ref), [`Syntomic.nabla_OK_column`](@ref), [`Syntomic.nabla_OK_matrix`](@ref) | `step5_nabla_ok.jl` | [AKN, §7.1, Step 5, p. 56] |
| 6 | `red`, `red^∇` | [`Syntomic.reduction_matrix`](@ref), [`Syntomic.reduction_matrix_nabla`](@ref) | `step6_reduction.jl` | [AKN, §7.1, Step 6, p. 56] |
| 7 | `∇` for `R` | [`Syntomic.nabla_R_matrix`](@ref) | `step7_nabla_r.jl` | [AKN, §7.1, Step 7, p. 56; Lemma 7.6, p. 59] |
| 8 | `N^{≥i}∇` | [`Syntomic.can_nabla_matrix`](@ref), [`Syntomic.nabla_N_matrix`](@ref) | `step8_nabla_n.jl`, `step10_can_phi_nabla.jl` | [AKN, §7.1, Step 8, p. 56; Lemma 7.8, p. 60] |
| 9 | `can - φ` | [`Syntomic.can_matrix`](@ref), [`Syntomic.phi_matrix`](@ref), [`Syntomic.can_minus_phi`](@ref) | `step9_can_phi.jl` | [AKN, §7.1, Step 9, p. 57] |
| 10 | `can - φ^∇` | [`Syntomic.phi_nabla_matrix`](@ref), [`Syntomic.can_minus_phi_nabla`](@ref) | `step10_can_phi_nabla.jl` | [AKN, §7.1, Step 10, p. 57; Lemma 7.9, p. 60] |
| Return | `syn^0`, `syn^1`, cohomology | [`Syntomic.syntomic_square`](@ref), [`syntomic_cohomology`](@ref), [`Syntomic.certified_elementary_divisors`](@ref), [`local_elementary_divisors`](@ref) | `pipeline.jl`, `cohomology.jl`, `linalg/local_snf.jl` | [AKN, §7.1, Return, p. 57; Lemma 7.1, p. 57; Remark 7.3, p. 58] |
| K-groups | `K_{2i-1}`, `K_{2i-2}` | [`k_groups`](@ref) | `kgroups.jl` | [AKN, Cor. 2.16, p. 9] |

```@docs
Syntomic.syntomic_wmatrices
Syntomic.syntomic_square
Syntomic.working_precision
Syntomic.exponent_phase
Syntomic.descent_ring
Syntomic.normalize_eisenstein
Syntomic.EisensteinData
Syntomic.target_precision
Syntomic.f_basis_exponents
Syntomic.g_basis_exponents
Syntomic.GaloisRing
Syntomic.zp_expand
Syntomic.zp_expand_descent
Syntomic.ws_frobenius
Syntomic.ws_delta
Syntomic.LambdaData
Syntomic.lambda_zero
Syntomic.lambda_denominator
Syntomic.lambda_next
Syntomic.FEnvelope
Syntomic.f_relation_coefficient
Syntomic.f_monomial_normal_form
Syntomic.f_mul
Syntomic.GEnvelope
Syntomic.g_Rprime_zero
Syntomic.g_Rprime_next
Syntomic.g_relation
Syntomic.g_frobenius_generator
Syntomic.g_monomial_normal_form
Syntomic.g_mul
Syntomic.g_frobenius
Syntomic.g_delta
Syntomic.divided_difference_E
Syntomic.phi_one_g0
Syntomic.bk_unit_u
Syntomic.bk_unit_v
Syntomic.bk_unit_power
Syntomic.nabla_theta_table
Syntomic.nabla_OK_column
Syntomic.nabla_OK_matrix
Syntomic.reduction_matrix
Syntomic.reduction_matrix_nabla
Syntomic.nabla_R_matrix
Syntomic.can_nabla_matrix
Syntomic.nabla_N_matrix
Syntomic.can_matrix
Syntomic.phi_matrix
Syntomic.can_minus_phi
Syntomic.phi_nabla_matrix
Syntomic.can_minus_phi_nabla
Syntomic.certified_elementary_divisors
```
