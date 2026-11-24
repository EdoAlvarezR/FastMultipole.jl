# NOTE: `curl_extras::Val` (whether to add the curl of the vector field and its
#       gradient to the extra outputs, see `evaluate_local_curl`) is the last
#       argument; the methods without it do not add them
evaluate_local!(system, i_system, tree::Tree, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches) =
    evaluate_local!(system, i_system, tree, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches, Val(false))

function evaluate_local!(system, i_system, tree::Tree, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches, curl_extras::Val)

    # loop over leaf branches
    for i_branch in tree.leaf_index
        evaluate_local!(system, i_system, tree, i_branch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches, curl_extras)
    end
end

function evaluate_local!(system, i_system, tree::Tree, i_branch::Integer, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches, curl_extras::Val=Val(false))
    branch = tree.branches[i_branch]
    local_expansion = view(tree.expansions, :, :, :, i_branch)
    evaluate_local!(system, branch.bodies_index[i_system], harmonics, gradient_n_m, local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switches[i_system], curl_extras)
end

# function evaluate_local!(systems::Tuple, branch::Branch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switches)
#     for (system, bodies_index, derivatives_switch) in zip(systems, branch.bodies_index, derivatives_switches)
#         evaluate_local!(system, bodies_index, harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switch)
#     end
# end

# function evaluate_local!(systems, branch::Branch{TF,<:Any}, expansion_order, lamb_helmholtz, derivatives_switches) where TF
#     harmonics = branch.harmonics
#     gradient_n_m = initialize_gradient_n_m(expansion_order, TF)
#     for (i, system) in enumerate(systems)
#         evaluate_local!(system, branch.bodies_index[i], harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switches[i])
#     end
# end

# function evaluate_local!(system, branch::SingleBranch, harmonics, gradient_n_m, expansion_order, lamb_helmholtz, derivatives_switch)
#     evaluate_local!(system, branch.bodies_index, harmonics, gradient_n_m, branch.local_expansion, branch.center, expansion_order, lamb_helmholtz, derivatives_switch)
# end

function evaluate_local!(system, bodies_index, harmonics, gradient_n_m, local_expansion, expansion_center, expansion_order, lamb_helmholtz, derivatives_switch::DerivativesSwitch{PS,GS,HS,NO}, ::Val{CE}=Val(false)) where {PS,GS,HS,NO,CE}
    for i_body in bodies_index
        pos = get_position(system, i_body)
        scalar_potential, gradient, hessian = evaluate_local(pos - expansion_center, harmonics, gradient_n_m, local_expansion, expansion_order, lamb_helmholtz, derivatives_switch)

        PS && set_scalar_potential!(system, derivatives_switch, i_body, scalar_potential)

        GS && set_gradient!(system, derivatives_switch, i_body, gradient)

        HS && set_hessian!(system, derivatives_switch, i_body, hessian)

        # Curl of the vector field and its gradient, added to the extra outputs 1:12
        # (needs the coefficients that `evaluate_local` leaves in `gradient_n_m`
        # when the hessian is requested)
        if CE && HS && NO >= 12 && lamb_helmholtz isa Val{true}
            curl, gradcurl = evaluate_local_curl(harmonics, gradient_n_m, expansion_order)
            for k in 1:3
                set_extra_output!(system, derivatives_switch, i_body, k, curl[k])
            end
            for k in 1:9
                set_extra_output!(system, derivatives_switch, i_body, 3 + k, gradcurl[k])
            end
        end
    end
end

# Coefficients of the derivative along x, y, and z of the harmonic series with
# coefficients C[:, c, :] (component c of `gradient_n_m`), at the term (n, m)
# with index i (the same index shifts as the vector gradient in `evaluate_local`)
@inline function _derivative_coefficients(C, c, i, n, m)
    @inbounds if m == 0
        dx = (-C[2,c,i+n+2], zero(eltype(C)))
        dy = (-C[1,c,i+n+2], zero(eltype(C)))
        dz = (-C[1,c,i+n+1], -C[2,c,i+n+1])
    else
        dx = (-(C[2,c,i+n] + C[2,c,i+n+2]) * 0.5, (C[1,c,i+n] + C[1,c,i+n+2]) * 0.5)
        dy = ((C[1,c,i+n] - C[1,c,i+n+2]) * 0.5, (C[2,c,i+n] - C[2,c,i+n+2]) * 0.5)
        dz = (-C[1,c,i+n+1], -C[2,c,i+n+1])
    end
    return dx, dy, dz
end

"""
    evaluate_local_curl(harmonics, gradient_n_m, expansion_order) -> curl, gradcurl

Curl ω = ∇×v of the vector field of a local expansion and its gradient
(`gradcurl[i, k] = ∂ω_i/∂x_k`, column major) at the point where the regular
harmonics `harmonics` were last evaluated, given the expansion of the vector
field in `gradient_n_m[:, 1:3, :]` as left by `evaluate_local` (with the
hessian requested). The expansion of the curl follows from that of the field
by one more index shift (derivatives of the regular harmonics are regular
harmonics of one degree less), and its gradient by another, so it needs no
other expansion. It is stored in `gradient_n_m[:, 4:6, :]`.
"""
function evaluate_local_curl(harmonics, gradient_n_m, expansion_order)

    TF = eltype(gradient_n_m)
    C = gradient_n_m

    # Expansion of the curl, for n ≤ p - 1
    i = 0
    for n in 0:expansion_order-1, m in 0:n
        i += 1
        (vx_x, vx_y, vx_z) = _derivative_coefficients(C, 1, i, n, m)
        (vy_x, vy_y, vy_z) = _derivative_coefficients(C, 2, i, n, m)
        (vz_x, vz_y, vz_z) = _derivative_coefficients(C, 3, i, n, m)
        @inbounds begin
            # ω_x = ∂_y v_z - ∂_z v_y, ω_y = ∂_z v_x - ∂_x v_z, ω_z = ∂_x v_y - ∂_y v_x
            C[1,4,i] = vz_y[1] - vy_z[1];   C[2,4,i] = vz_y[2] - vy_z[2]
            C[1,5,i] = vx_z[1] - vz_x[1];   C[2,5,i] = vx_z[2] - vz_x[2]
            C[1,6,i] = vy_x[1] - vx_y[1];   C[2,6,i] = vy_x[2] - vx_y[2]
        end
    end

    # Evaluate the curl (n ≤ p - 1) and its gradient (n ≤ p - 2)
    w = MVector{3,TF}(0, 0, 0)
    gw = MMatrix{3,3,TF,9}(0, 0, 0, 0, 0, 0, 0, 0, 0)
    i = 0
    for n in 0:expansion_order-1, m in 0:n
        i += 1
        f = m == 0 ? one(TF) : 2*one(TF)
        @inbounds Rr, Ri = harmonics[1,1,i], harmonics[2,1,i]
        for c in 1:3
            @inbounds w[c] += f * (C[1,3+c,i] * Rr - C[2,3+c,i] * Ri)
            if n <= expansion_order - 2
                (dx, dy, dz) = _derivative_coefficients(C, 3+c, i, n, m)
                gw[c,1] += f * (dx[1] * Rr - dx[2] * Ri)
                gw[c,2] += f * (dy[1] * Rr - dy[2] * Ri)
                gw[c,3] += f * (dz[1] * Rr - dz[2] * Ri)
            end
        end
    end

    return SVector(w) * ONE_OVER_4π, SMatrix(gw) * ONE_OVER_4π
end

function evaluate_local(Δx, harmonics, gradient_n_m, local_expansion, expansion_order, ::Val{LH}, ::DerivativesSwitch{PS,GS,HS}) where {LH,PS,GS,HS}
    # convert to spherical coordinates
    r, θ, ϕ = cartesian_to_spherical(Δx)

    # expansion basis is the regular solid harmonics
    regular_harmonics!(harmonics, r, θ, ϕ, expansion_order)

    #--- declare/reset variables ---#

    # scalar potential
    u = zero(eltype(local_expansion))

    # vector field
    vx, vy, vz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    HS && ( gradient_n_m .= zero(eltype(local_expansion)) )

    # vector gradient
    vxx, vxy, vxz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    vyx, vyy, vyz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))
    vzx, vzy, vzz = zero(eltype(local_expansion)), zero(eltype(local_expansion)), zero(eltype(local_expansion))

    # index
    i_n_m = 0

    #------- n = 0, m = 0 -------#

    n = 0

    # update index
    i_n_m += 1

    # get regular harmonic
    Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

    # scalar potential
    if PS # && !LH # scalar potential becomes non-sensical to preserve vector field when using Lamb-Helmholtz
        ϕ_n_m_real = local_expansion[1,1,i_n_m]
        ϕ_n_m_imag = local_expansion[2,1,i_n_m]
        u += Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag
    end

    # vector
    if GS || HS
        #=
        vx = -Im[ϕ_{1}^{1}]
        vy = -Re[ϕ_{1}^{1}]
        vz = -ϕ_{1}^0
        =#

        # expansion coefficients
        vx_n_m_real = zero(eltype(local_expansion))
        vx_n_m_imag = zero(eltype(local_expansion))
        vy_n_m_real = zero(eltype(local_expansion))
        vy_n_m_imag = zero(eltype(local_expansion))
        vz_n_m_real = zero(eltype(local_expansion))
        vz_n_m_imag = zero(eltype(local_expansion))

        # due to ϕ
        ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
        ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
        ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
        ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

        vx_n_m_real -= ϕ_np1_mp1_imag
        vy_n_m_real -= ϕ_np1_mp1_real
        vz_n_m_real -= ϕ_np1_m_real
        vz_n_m_imag -= ϕ_np1_m_imag

        if GS
            # evaluate expansion
            vx += vx_n_m_real * Rnm_real
            vy += vy_n_m_real * Rnm_real
            vz += vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag
        end

        if HS
            # store components for computing vector gradient
            gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
            gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
            gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
            gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
        end

    end

    #------- n > 0 -------#

    for n in 1:expansion_order

        #--- m = 0 ---#

        # update index
        i_n_m += 1

        # get regular harmonic
        Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

        # scalar potential
        if PS && !LH
            ϕ_n_m_real = local_expansion[1,1,i_n_m]
            ϕ_n_m_imag = local_expansion[2,1,i_n_m]
            u += Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag
        end

        # vector
        if GS || HS
            #=
            vx = (Re[χ_1^1] - Im[ϕ_2^1]) R_n^m
            vy = -(Re[ϕ_2^1] + Im[χ_1^1]) R_n^m
            vz = -ϕ_2^0 R_n^m
            =#

            # expansion coefficients
            vx_n_m_real = zero(eltype(local_expansion))
            vx_n_m_imag = zero(eltype(local_expansion))
            vy_n_m_real = zero(eltype(local_expansion))
            vy_n_m_imag = zero(eltype(local_expansion))
            vz_n_m_real = zero(eltype(local_expansion))
            vz_n_m_imag = zero(eltype(local_expansion))

            # due to ϕ
            if n < expansion_order
                ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
                ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
                ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
                ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

                vx_n_m_real -= ϕ_np1_mp1_imag
                vy_n_m_real -= ϕ_np1_mp1_real
                vz_n_m_real -= ϕ_np1_m_real
                vz_n_m_imag -= ϕ_np1_m_imag
            end


            # due to χ
            if LH
                χ_n_mp1_real = local_expansion[1,2,i_n_m+1]
                χ_n_mp1_imag = local_expansion[2,2,i_n_m+1]

                vx_n_m_real += n * χ_n_mp1_real
                vy_n_m_real -= n * χ_n_mp1_imag
            end

            if GS
                # evaluate expansion
                vx += vx_n_m_real * Rnm_real
                vy += vy_n_m_real * Rnm_real
                vz += vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag
            end

            if HS
                # store components for computing vector gradient
                gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
                gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
                gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
                gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
            end

        end

        #--- m > 0 ---#

        for m in 1:n

            # update index
            i_n_m += 1

            # get regular harmonic
            Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

            # scalar potential
            if PS && !LH
                ϕ_n_m_real = local_expansion[1,1,i_n_m]
                ϕ_n_m_imag = local_expansion[2,1,i_n_m]
                u += 2 * (Rnm_real * ϕ_n_m_real - Rnm_imag * ϕ_n_m_imag)
            end

            # vector
            if GS || HS
                #=
                vx = (im * (ϕ_{n+1}^{m-1} + ϕ_{n+1}^{m+1}) + (n-m) χ_n^{m+1} - (n+m) χ_n^{m-1}) / 2
                vy = (ϕ_{n+1}^{m-1} - ϕ_{n+1}^{m+1} + im * (n-m) * χ_n^{m+1} + im * (n+m) χ_n^{m-1}) / 2
                vz = -ϕ_{n+1}^m - im * m * χ_n^m
                =#

                # expansion coefficients
                vx_n_m_real = zero(eltype(local_expansion))
                vx_n_m_imag = zero(eltype(local_expansion))
                vy_n_m_real = zero(eltype(local_expansion))
                vy_n_m_imag = zero(eltype(local_expansion))
                vz_n_m_real = zero(eltype(local_expansion))
                vz_n_m_imag = zero(eltype(local_expansion))

                if n < expansion_order
                    ϕ_np1_mm1_real = local_expansion[1,1,i_n_m+n]
                    ϕ_np1_mm1_imag = local_expansion[2,1,i_n_m+n]
                    ϕ_np1_m_real = local_expansion[1,1,i_n_m+n+1]
                    ϕ_np1_m_imag = local_expansion[2,1,i_n_m+n+1]
                    ϕ_np1_mp1_real = local_expansion[1,1,i_n_m+n+2]
                    ϕ_np1_mp1_imag = local_expansion[2,1,i_n_m+n+2]

                    vx_n_m_real -= (ϕ_np1_mm1_imag + ϕ_np1_mp1_imag) * 0.5
                    vx_n_m_imag += (ϕ_np1_mm1_real + ϕ_np1_mp1_real) * 0.5
                    vy_n_m_real += (ϕ_np1_mm1_real - ϕ_np1_mp1_real) * 0.5
                    vy_n_m_imag += (ϕ_np1_mm1_imag - ϕ_np1_mp1_imag) * 0.5
                    vz_n_m_real -= ϕ_np1_m_real
                    vz_n_m_imag -= ϕ_np1_m_imag
                end

                if LH
                    # extract expansion coefficients
                    χ_n_mm1_real = local_expansion[1,2,i_n_m-1]
                    χ_n_mm1_imag = local_expansion[2,2,i_n_m-1]

                    # form vector coefficients
                    vx_n_m_real -= (n+m) * χ_n_mm1_real * 0.5
                    vx_n_m_imag -= (n+m) * χ_n_mm1_imag * 0.5
                    vy_n_m_real -= (n+m) * χ_n_mm1_imag * 0.5
                    vy_n_m_imag += (n+m) * χ_n_mm1_real * 0.5
                    if m < n
                        χ_n_mp1_real = local_expansion[1,2,i_n_m+1]
                        χ_n_mp1_imag = local_expansion[2,2,i_n_m+1]

                        vx_n_m_real += (n-m) * χ_n_mp1_real * 0.5
                        vx_n_m_imag += (n-m) * χ_n_mp1_imag * 0.5
                        vy_n_m_real -= (n-m) * χ_n_mp1_imag * 0.5
                        vy_n_m_imag += (n-m) * χ_n_mp1_real * 0.5
                    end

                    # extract expansion coefficients
                    χ_n_m_real = local_expansion[1,2,i_n_m]
                    χ_n_m_imag = local_expansion[2,2,i_n_m]

                    # form vector coefficients
                    vz_n_m_real += m * χ_n_m_imag
                    vz_n_m_imag -= m * χ_n_m_real
                end

                # evaluate expansion
                vx += 2 * (vx_n_m_real * Rnm_real - vx_n_m_imag * Rnm_imag)
                vy += 2 * (vy_n_m_real * Rnm_real - vy_n_m_imag * Rnm_imag)
                vz += 2 * (vz_n_m_real * Rnm_real - vz_n_m_imag * Rnm_imag)

                if HS
                    # store components for computing vector gradient
                    gradient_n_m[1,1,i_n_m] = vx_n_m_real # x component
                    gradient_n_m[2,1,i_n_m] = vx_n_m_imag # x component
                    gradient_n_m[1,2,i_n_m] = vy_n_m_real # y component
                    gradient_n_m[2,2,i_n_m] = vy_n_m_imag # y component
                    gradient_n_m[1,3,i_n_m] = vz_n_m_real # z component
                    gradient_n_m[2,3,i_n_m] = vz_n_m_imag # z component
                end

            end
        end
    end

    if HS

        # index
        i_n_m = 0

        for n in 0:expansion_order-1

            #--- m = 0 ---#

            # update index
            i_n_m += 1

            # get regular harmonic
            Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

            vg_xx_real = -gradient_n_m[2,1,i_n_m+n+2]
            vxx += vg_xx_real * Rnm_real

            vg_yx_real = -gradient_n_m[1,1,i_n_m+n+2]
            vyx += vg_yx_real * Rnm_real

            vg_zx_real = -gradient_n_m[1,1,i_n_m+n+1]
            vg_zx_imag = -gradient_n_m[2,1,i_n_m+n+1]
            vzx += vg_zx_real * Rnm_real - vg_zx_imag * Rnm_imag

            vg_xy_real = -gradient_n_m[2,2,i_n_m+n+2]
            vxy += vg_xy_real * Rnm_real

            vg_yy_real = -gradient_n_m[1,2,i_n_m+n+2]
            vyy += vg_yy_real * Rnm_real

            vg_zy_real = -gradient_n_m[1,2,i_n_m+n+1]
            vg_zy_imag = -gradient_n_m[2,2,i_n_m+n+1]
            vzy += vg_zy_real * Rnm_real - vg_zy_imag * Rnm_imag

            vg_xz_real = -gradient_n_m[2,3,i_n_m+n+2]
            vxz += vg_xz_real * Rnm_real

            vg_yz_real = -gradient_n_m[1,3,i_n_m+n+2]
            vyz += vg_yz_real * Rnm_real

            vg_zz_real = -gradient_n_m[1,3,i_n_m+n+1]
            vg_zz_imag = -gradient_n_m[2,3,i_n_m+n+1]
            vzz += vg_zz_real * Rnm_real - vg_zz_imag * Rnm_imag

            #--- m > 0 ---#

            for m in 1:n

                # update index
                i_n_m += 1

                # get regular harmonic
                Rnm_real, Rnm_imag = harmonics[1,1,i_n_m], harmonics[2,1,i_n_m]

                vg_xx_real = -(gradient_n_m[2,1,i_n_m+n] + gradient_n_m[2,1,i_n_m+n+2]) * 0.5
                vg_xx_imag = (gradient_n_m[1,1,i_n_m+n] + gradient_n_m[1,1,i_n_m+n+2]) * 0.5
                vxx += 2 * (vg_xx_real * Rnm_real - vg_xx_imag * Rnm_imag)

                vg_yx_real = (gradient_n_m[1,1,i_n_m+n] - gradient_n_m[1,1,i_n_m+n+2]) * 0.5
                vg_yx_imag = (gradient_n_m[2,1,i_n_m+n] - gradient_n_m[2,1,i_n_m+n+2]) * 0.5
                vyx += 2 * (vg_yx_real * Rnm_real - vg_yx_imag * Rnm_imag)

                vg_zx_real = -gradient_n_m[1,1,i_n_m+n+1]
                vg_zx_imag = -gradient_n_m[2,1,i_n_m+n+1]
                vzx += 2 * (vg_zx_real * Rnm_real - vg_zx_imag * Rnm_imag)

                vg_xy_real = -(gradient_n_m[2,2,i_n_m+n] + gradient_n_m[2,2,i_n_m+n+2]) * 0.5
                vg_xy_imag = (gradient_n_m[1,2,i_n_m+n] + gradient_n_m[1,2,i_n_m+n+2]) * 0.5
                vxy += 2 * (vg_xy_real * Rnm_real - vg_xy_imag * Rnm_imag)

                vg_yy_real = (gradient_n_m[1,2,i_n_m+n] - gradient_n_m[1,2,i_n_m+n+2]) * 0.5
                vg_yy_imag = (gradient_n_m[2,2,i_n_m+n] - gradient_n_m[2,2,i_n_m+n+2]) * 0.5
                vyy += 2 * (vg_yy_real * Rnm_real - vg_yy_imag * Rnm_imag)

                vg_zy_real = -gradient_n_m[1,2,i_n_m+n+1]
                vg_zy_imag = -gradient_n_m[2,2,i_n_m+n+1]
                vzy += 2 * (vg_zy_real * Rnm_real - vg_zy_imag * Rnm_imag)

                vg_xz_real = -(gradient_n_m[2,3,i_n_m+n] + gradient_n_m[2,3,i_n_m+n+2]) * 0.5
                vg_xz_imag = (gradient_n_m[1,3,i_n_m+n] + gradient_n_m[1,3,i_n_m+n+2]) * 0.5
                vxz += 2 * (vg_xz_real * Rnm_real - vg_xz_imag * Rnm_imag)

                vg_yz_real = (gradient_n_m[1,3,i_n_m+n] - gradient_n_m[1,3,i_n_m+n+2]) * 0.5
                vg_yz_imag = (gradient_n_m[2,3,i_n_m+n] - gradient_n_m[2,3,i_n_m+n+2]) * 0.5
                vyz += 2 * (vg_yz_real * Rnm_real - vg_yz_imag * Rnm_imag)

                vg_zz_real = -gradient_n_m[1,3,i_n_m+n+1]
                vg_zz_imag = -gradient_n_m[2,3,i_n_m+n+1]
                vzz += 2 * (vg_zz_real * Rnm_real - vg_zz_imag * Rnm_imag)

            end
        end
    end

    return u * ONE_OVER_4π, SVector{3}(vx,vy,vz) * ONE_OVER_4π, SMatrix{3,3,eltype(local_expansion),9}(vxx, vxy, vxz, vyx, vyy, vyz, vzx, vzy, vzz) * ONE_OVER_4π
end
