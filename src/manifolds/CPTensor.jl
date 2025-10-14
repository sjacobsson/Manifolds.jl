# using LowRankApprox # bodge
#
# TODO: Rename CanonicalPolyadic
# TODO: Warning when r is too large
using TensorToolbox

@doc raw"""
    CPTensor{𝔽,V,R} <: AbstractManifold{𝔽}

The manifold of tensors with CP rank ``R`` and multilinear rank ``(R, ..., R)``.
"""
struct CPTensor{𝔽,V,R} <: AbstractManifold{𝔽} end

function CPTensor(valence::NTuple{D, Int},rank::Int; field::AbstractNumbers=ℝ) where D
    @assert (all(rank .<= valence)) "Manifold not implemented when rank is larger than valence."

    return CPTensor{field,valence,rank}()
end

function check_size(M::CPTensor{𝔽,V,R}, p) where {𝔽,V,R}#={{{=#
    p_size = size.(p)
    M_size = [(v, R) for v in V]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end#=}}}=#

function check_size(M::CPTensor{𝔽,V,R}, p, X;) where {𝔽,V,R}#={{{=#
    p_size = size.(p)
    X_size = size.(X)
    M_size = [(v, R) for v in V]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $(X) can not belong to the manifold $(M), since its size $(X_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end#=}}}=#

# TODO: Only define embed!
@doc raw"""
    TODO
"""
function embed(M::CPTensor{𝔽,V,R}, p) where {𝔽,V,R}#={{{=#

    return full(ktensor(p))
end#=}}}=#

function embed!(M::CPTensor{𝔽,V,R}, q, p) where {𝔽,V,R}#={{{=#

    return q .= sum([
        kron([p[i][:, k] for i in 1:length(V)]...)
        for k in 1:R
    ])
end#=}}}=#

@doc raw"""
    TODO
"""
is_point(M::CPTensor{ℝ,V,R}, p; kwargs...) where {V,R}

function check_point(M::CPTensor{ℝ,V,R}, p; atol=1.4901161193847656e-8, kwargs...) where {V,R}#={{{=#

    # TODO: What about the permutations?
    for (g, n) in zip(p, V)
        e = check_point(InvertibleMatrices(R), g' * g; atol=atol, kwargs...)
        if !isnothing(e)
            return e
        end
    end
end#=}}}=#

@doc raw"""
    TODO
"""
is_vector(M::CPTensor{ℝ,V,R}, p, v; kwargs...) where {V,R}

function check_vector(M::CPTensor{ℝ,V,R}, p, X; atol=1.4901161193847656e-8, kwargs...) where {V,R}#={{{=#

    ds = [] # Initialize
    for (i, n) in enumerate(V)
        g11 = p[i][1:R, :]
        g21 = p[i][R + 1:end, :]
        tmp1 = g21 / g11
        tmp2 = tmp1' * tmp1
        Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'

        push!(ds, diag(p[i]' * (X[i] * (I - Gamma12 * g21) / (g11' * g11))))
    end

    if !all([isapprox(d, ds[1]; atol=atol) for d in ds])
        return DomainError(X, "$(X) is not orthogonal to the fiber at $(p).")
    end
    # end
end#=}}}=#

# Estimate (exp(A) - 1) / A.
# See Functions of Matrices (Higham 2008) section 10.7.4
function psi1(A)#={{{=#
    z = ceil(log(2, norm(A))) + 2

    if z <= 0
        A2 = A * A
        A3 = A2 * A
        A4 = A3 * A
        A5 = A4 * A
        A6 = A5 * A

        return (I + A / 26 + 5 * A2 / 156 + A3 / 858 + A4 / 5720 + A5 / 205920 + A6 / 8648640) /
            (I - 6 * A / 13 + 5 * A2 / 52 - 5 * A3 / 429 + A4 / 1144 - A5 / 25740 + A6 / 1235520)
    else
        B = A / 2^z
        B2 = B * B
        B3 = B2 * B
        B4 = B3 * B
        B5 = B4 * B
        B6 = B5 * B

        fB = (I + B / 26 + 5 * B2 / 156 + B3 / 858 + B4 / 5720 + B5 / 205920 + B6 / 8648640) /
            (I - 6 * B / 13 + 5 * B2 / 52 - 5 * B3 / 429 + B4 / 1144 - B5 / 25740 + B6 / 1235520)
        mexpB = (I + B / 2 + 5 * B2 / 44 + B3 / 66 + B4 / 792 + B5 / 15840 + B6 / 665280) /
            (I - B / 2 + 5 * B2 / 44 - B3 / 66 + B4 / 792 - B5 / 15840 + B6 / 665280)

        out = fB / 2^z # Initialize
        tmp = 1 * mexpB
        for _ in 1:z
            out = out * (tmp + I)
            tmp = tmp * tmp
        end

        return out
    end
end#=}}}=#

# Helper function to invert permutation, using "word notation":
#   https://en.wikipedia.org/wiki/Permutation#One-line_notation
# Adapted from Permutations.jl:
#   https://www.juliapackages.com/p/permutations
function invperm(p::Vector{Int64})#={{{=#
    n = length(p)
    data = zeros(Int, n)
    for k = 1:n
        @inbounds j = p[k]
        @inbounds data[j] = k
    end
    return data
end#=}}}=#

@doc raw"""
    TODO
"""
exp(M::CPTensor{ℝ,V,R}, p, X) where {V,R}

function exp!(M::CPTensor{ℝ,V,R}, q, p, X) where {V,R}#={{{=#

    # Permute rows so that upper left block has max volume
    _p = similar.(p) # Initialize
    _X = similar.(X) # Initialize
    permutations = [] # Initialize
    for (i, n) in enumerate(V)
        rows, _ = cur(p[i])
        permutation = [rows..., [i for i in 1:n if !in(i, rows)]...]
        push!(permutations, permutation)
        _p[i] .= p[i][permutation, :]
        _X[i] .= X[i][permutation, :]
    end

    _q = similar.(p) # Initialize
    for (i, n) in enumerate(V)
        g11 = _p[i][1:R, :]
        g21 = _p[i][R + 1:end, :]
        X11 = _X[i][1:R, :]
        X21 = _X[i][R + 1:end, :]

        tmp1 = g21 / g11
        tmp2 = tmp1' * tmp1
        Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'

        A = [X11; X21]
        B = [((I - Gamma12 * g21) / g11) Gamma12]
        Aprim = [X11 (-(I - Gamma12 * g21) / g11)'; X21 (-Gamma12')]
        Bprim = [((I - Gamma12 * g21) / g11) Gamma12; X11' X21']

        g = _p[i]
        psiBA = psi1(B * A)
        psiBprimAprim = psi1(Bprim * Aprim)
        _q[i] .=
            g +
            Aprim * psiBprimAprim * (Bprim * g) +
            B' * (psiBA' * (A' * g)) +
            Aprim * psiBprimAprim * (Bprim * B') * (psiBA' * (A' * g))
    end

    # Permute back rows to how they were before
    for i in 1:length(V)
        permutation = permutations[i]
        q[i] .= _q[i][invperm(permutation), :]
    end

    return q

    # Simple necessary conditions for being geodesic. TODO: Write actual tests.
    # The derivative of exp(t X) is tangent at exp(t X):
    #   using Manifolds, FiniteDifferences; M = CPTensor((3, 3, 4), 2); p = rand(M); X = rand(M, vector_at=p); println(is_vector(M, exp(M, p, X), central_fdm(3, 1; adapt=0)(t -> p - exp(M, p, t * X), 1.0); atol=1e-2))
    # The derivative of exp(t X) has same norm as X:
    #   using Manifolds, FiniteDifferences; M = CPTensor((3, 3, 4), 2); p = rand(M); X = rand(M, vector_at=p); println(norm(M, p, X), " ≈ ", norm(M, p, central_fdm(3, 1; adapt=0)(t -> p - exp(M, p, t * X), 0.0)))
    # The first derivative of the length functional is 0, and the second derivative is positive:
    #   using Manifolds, Trapz, FiniteDifferences; M = CPTensor((3, 3, 4), 2); p = rand(M); X = rand(M, vector_at=p); Y = rand(M, vector_at=p); X0 = 0.0 * X; X1 = 0.5 * X; X2 = 1.0 * X; b(t, s) = (1 - t)^2 * X0 + 2 * (1 - t) * t * (X1 + s * Y) + t^2 * X2 #= Bezier curve from X0 to X2 with control point X1 + s * Y =#; gamma(t, s) = exp(M, p, b(t, s)); dt = 1e-3; ts = 0.0:dt:1.0; curve_length(s) = trapz(ts, [norm(M, gamma(t, s), central_fdm(3, 1; adapt=0)(r -> gamma(r, s), t)) for t in ts]); println(central_fdm(3, 1)(curve_length, 0.0), " ≈ 0"); println(central_fdm(3, 2)(curve_length, 0.0), " > 0")
end#=}}}=#

@doc raw"""
    inner(M::CPTensor{𝔽,V,R}, p, X, Y,)
"""
function inner(M::CPTensor{𝔽,V,R}, p, X, Y) where {𝔽,V,R}#={{{=#

    s = 0 # Initialize
    for (i, n) in enumerate(V)
        g11 = p[i][1:R, :]
        g21 = p[i][R + 1:end, :]
        tmp1 = g21 / g11
        tmp2 = tmp1' * tmp1
        Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'

        tmp = (I - Gamma12 * g21) / g11
        s = s + tr((Y[i]' * X[i]) * (tmp * tmp' + Gamma12 * Gamma12'))
    end

    return s
end#=}}}=#

@doc raw"""
    TODO
"""
function project(M::CPTensor{𝔽,V,R}, p, A) where {𝔽,V,R}
    # TODO: project rank 1 tensor A to T_p M
    # Implement this using minres or gmres from IterativeSolvers.jl
end

@doc raw"""
    zero_vector(M::CPTensor{𝔽,V,R}, p)
"""
function zero_vector(M::CPTensor{𝔽,V,R}, p) where {𝔽,V,R}#={{{=#

    return [zeros(n, R) for n in V]
end#=}}}=#

manifold_dimension(::CPTensor{𝔽,V,R}) where {𝔽,V,R} = 0 # TODO

@doc raw"""
    rand(M::CPTensor{ℝ,V,R}; vector_at=nothing)
"""
function rand(M::CPTensor{ℝ,V,R}; vector_at=nothing, kwargs...) where {V,R}#={{{=#
    if isnothing(vector_at)
        return [vcat(rand(InvertibleMatrices(R)), rand(n - R, R)) for n in V]
    else
        X = similar.(vector_at) # Initialize
        _X = [rand(s1, s2) for (s1, s2) in size.(vector_at)]

        # Enforce the following condition: The kth column of
        #   [X11; X21] * (1 - Gamma12 * g21) * inv(g11) * inv(g11)'
        # has the same dot product with the kth column of [g11; g21] for every i.
        for (i, n) in enumerate(V)
            g11 = vector_at[i][1:R, :]
            g21 = vector_at[i][R + 1:end, :]
            tmp1 = g21 / g11
            tmp2 = tmp1' * tmp1
            Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'

            for k in 1:R
                g1k = vector_at[1][:, k]
                X1k = _X[1][:, k]
                gik = vector_at[i][:, k]
                Xik = _X[i][:, k]
                _X[i][:, k] .= Xik - gik * dot(gik, Xik) / dot(gik, gik) + gik * dot(g1k, X1k) / dot(gik, gik)
            end
            X[i] .= _X[i] * g11' * g11 / (I - Gamma12 * g21)
        end

        return X
    end
end#=}}}=#


function Base.show(io::IO, ::CPTensor{𝔽,V,R}) where {𝔽,V,R}
    return print(io, "CPTensor(($(join(V, ", "))), $(R); field=$(𝔽))")
end
