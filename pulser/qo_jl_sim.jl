using QuantumOptics
using LinearAlgebra
using Interpolations
using Combinatorics

function tensor_basis(b, N)
    return b^N
end

function tensor_list(list)
    result = list[1]
    for x in list[2:length(list)]
        result = tensor(result, x)
    end
    return result
end

function build_vdw(qdict, sigma_rr, U, qid_index, b, N)
    comb = combinations(collect(keys(qdict)), 2)
    terms = [(U / norm(qdict[q1] - qdict[q2])^6) * embed(b, [N - qid_index[q1], N - qid_index[q2]], [sigma_rr, sigma_rr]) for (q1, q2) ∈ comb]
    return Base.sum(terms)
end

function pulser_schroedinger(tspan, times_interp, psi0, terms, vdw, b)
    function interp_terms()
        """
            Interpolate the given coefficients to plug them into the
            schroedinger simulation
        """
        step = times_interp[2] - times_interp[1]
        range = times_interp[1]:step:(times_interp[end])
        range2 = times_interp[1]:step:(times_interp[end] - step)
        # we remove one step at the end to account for rounding errors in
        # waveform preparations
        itp_terms = [length(range) == length(c) ? scale(interpolate(c, BSpline(Cubic(Line(OnGrid())))), range) : scale(interpolate(c, BSpline(Cubic(Line(OnGrid())))), range2) for (o, c) ∈ terms]
        return itp_terms
    end
    coeffs = interp_terms()
    Hterms = [o for (o, c) ∈ terms]
    # hermitian conjugates
    append!(Hterms, [copy(dagger(o)) for o ∈ Hterms])
    coeffs_cache = [c(tspan[1]) for c ∈ coeffs]
    append!(coeffs_cache, [conj(x) for x ∈ coeffs_cache])
    H_cache = LazySum(coeffs_cache, Hterms)
    Htot = LazySum(H_cache, vdw)
    function H!(t,psi)
        @inbounds for i=1:length(coeffs)
            c = coeffs[i](t)
            H_cache.factors[i] = c
            H_cache.factors[i + length(coeffs)] = conj(c)
        end
        return Htot
    end
    return timeevolution.schroedinger_dynamic(tspan, psi0, H!)
end
