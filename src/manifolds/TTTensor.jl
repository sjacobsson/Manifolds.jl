using LowRankApprox # bodge
using Kronecker: kronecker # TODO: consider using tkron from TensorToolbox
using TensorToolbox

@doc raw"""
    TTTensor{𝔽,V,S} <: AbstractManifold{𝔽}

The manifold of tensors with TT rank ``S = (s1, ..., s{d-1})`` and multilinear rank ``(s1, s1 * s2, ..., s{d - 2} * s{d - 1}, s{d - 1})``.
"""
struct TTTensor{𝔽,V,S} <: AbstractManifold{𝔽} end

function TTTensor(valence::NTuple{D, Int}, ranks::NTuple{D_, Int}; field::AbstractNumbers=ℝ) where {D, D_}
    # @assert(all(rank .<= valence)) TODO: adapt from CP to TT
    # mrank is [1, S...] .* [S..., 1]
    
    @assert D_ == D - 1 "Length of TT rank must be one less than the valence."

    return TTTensor{field,valence,ranks}()
end

function check_size(M::TTTensor{𝔽,V,S}, p) where {𝔽,V,S}#={{{=#
    p_size = size.(p)
    M_size = [(v, s) for (v, s) in zip(V, [1, S...] .* [S..., 1])]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end#=}}}=#

function check_size(M::TTTensor{𝔽,V,S}, p, X;) where {𝔽,V,S}#={{{=#
    p_size = size.(p)
    X_size = size.(X)
    M_size = [(v, s) for (v, s) in zip(V, [1, S...] .* [S..., 1])]

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
function embed(M::TTTensor{𝔽,V,S}, p) where {𝔽,V,S}#={{{=#
    Cs = CoreCell(undef, length(V)) # Initialize
    for (i, n) in enumerate(V)
        s = [1, S...][i]
        s_ = [S..., 1][i]

        Cs[i] = matten(p[i], 2, [s, n, s_])
    end

    return full(TTtensor(Cs))
end#=}}}=#

# function embed!(M::TTTensor{𝔽,V,S}, q, p) where {𝔽,V,S}#={{{=#
    
#     return q .= sum([
#         kron([p[i][:, k] for i in 1:length(V)]...)
#         for k in 1:S
#     ])
# end#=}}}=#

@doc raw"""
    TODO
"""
is_point(M::TTTensor{ℝ,V,S}, p; kwargs...) where {V,S}

function check_point(M::TTTensor{ℝ,V,S}, p; atol=1.4901161193847656e-8, kwargs...) where {V,S}#={{{=#
    # TODO
end#=}}}=#

@doc raw"""
    TODO
"""
is_vector(M::TTTensor{ℝ,V,S}, p, v; kwargs...) where {V,S}

function check_vector(M::TTTensor{ℝ,V,S}, p, X; atol=1.4901161193847656e-8, kwargs...) where {V,S}#={{{=#
    Ys = [] # Initialize

    # Get ONB for vertical space
    for (i, s) in enumerate(S); for a in 1:s; for b in 1:s
        Y = similar.(X) # Initialize

        Ks = [zeros(s, s) for s in S]
        Ks[i][a, b] = 1.0

        for (i_, n) in enumerate(V)
            s = [1, S...][i_]
            s_ = [S..., 1][i_]
            K = [Matrix(0I, 1, 1), Ks...][i_]
            K_ = [Ks..., Matrix(0I, 1, 1)][i_]

            # Vertical vector
            g11 = p[i_][1:s * s_, :]
            g21 = p[i_][s * s_ + 1:end, :]
            tmp1 = g21 / g11
            tmp2 = tmp1' * tmp1
            Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'
            Y[i_] .= p[i_] * 
                (-kronecker(K', Matrix(I, s_, s_)) + kronecker(Matrix(I, s, s), K_)) *
                ((I - Gamma12 * g21) / (g11' * g11))'
        end

        # Gram-Schmidt
        for Y_ in Ys
            Y .= Y - dot(Y, Y_) * Y_ / dot(Y_, Y_)
        end
        Ys = [Ys..., Y]

    end; end; end

    # Check projection to vertical space
    e_squared = 0 # Initialize
    for Y in Ys
        e_squared = e_squared + (abs(dot(X, Y)) / norm(Y))^2
    end
    
    if (sqrt(e_squared) > atol)
        return DomainError(X, "$(X) is not orthogonal to the fiber at $(p).")
    end
end#=}}}=#


@doc raw"""
    TODO
"""
exp(M::TTTensor{ℝ,V,S}, p, X) where {V,S}

function exp!(M::TTTensor{ℝ,V,S}, q, p, X; reorthogonalize=true) where {V,S}#={{{=#
    
    # BODGE
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


    # BODGE
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

    # Permute rows so that upper block has max volume
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
        s = [1, S...][i]
        s_ = [S..., 1][i]

        g11 = _p[i][1:s * s_, :]
        g21 = _p[i][s * s_ + 1:end, :]
        X11 = _X[i][1:s * s_, :]
        X21 = _X[i][s * s_ + 1:end, :]
        
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

    if reorthogonalize
        Cs = CoreCell(undef, length(V)) # Initialize
        for (i, n) in enumerate(V)
            s = [1, S...][i]
            s_ = [S..., 1][i]

            Cs[i] = matten(q[i], 2, [s, n, s_])
        end

        Cs = reorth(TTtensor(Cs)).cores

        for (i, n) in enumerate(V)
            s = [1, S...][i]
            s_ = [S..., 1][i]

            q[i] .= tenmat(Cs[i], 2)
        end
    end

    return q

    # Simple necessary conditions for being geodesic. TODO: Write actual tests.
    # The derivative of exp(t X) is tangent at exp(t X):
    #   using Manifolds, FiniteDifferences; V = (5, 5, 5); S = (2, 2); M = TTTensor(V, S); p = rand(M); X = rand(M, vector_at=p); println(is_vector(M, exp(M, p, X), central_fdm(3, 1; adapt=0)(t -> p - exp(M, p, t * X), 1.0); atol=1e-2))
    # The derivative of exp(t X) has same norm as X:
    #   using Manifolds, FiniteDifferences; V = (5, 5, 5); S = (2, 2); M = TTTensor(V, S); p = rand(M); X = rand(M, vector_at=p); println(norm(M, p, X), " ≈ ", norm(M, p, central_fdm(3, 1; adapt=0)(t -> p - exp(M, p, t * X), 0.0)))
    # The first derivative of the length functional is 0, and the second derivative is positive:
    #   using Manifolds, Trapz, FiniteDifferences; V = (5, 5, 5); S = (2, 2); M = TTTensor(V, S); p = rand(M); X = rand(M, vector_at=p); Y = rand(M, vector_at=p); X0 = 0.0 * X; X1 = 0.5 * X; X2 = 1.0 * X; b(t, s) = (1 - t)^2 * X0 + 2 * (1 - t) * t * (X1 + s * Y) + t^2 * X2 #= Bezier curve from X0 to X2 with control point X1 + s * Y =#; gamma(t, s) = exp(M, p, b(t, s)); dt = 1e-3; ts = 0.0:dt:1.0; curve_length(s) = trapz(ts, [norm(M, gamma(t, s), central_fdm(3, 1; adapt=0)(r -> gamma(r, s), t)) for t in ts]); println(central_fdm(3, 1)(curve_length, 0.0), " ≈ 0"); println(central_fdm(3, 2)(curve_length, 0.0), " > 0")
end#=}}}=#

@doc raw"""
    inner(M::TTTensor{𝔽,V,S}, p, X, Y,)
"""
function inner(M::TTTensor{𝔽,V,S}, p, X, Y) where {𝔽,V,S}#={{{=#

    summ = 0 # Initialize
    for (i, n) in enumerate(V)
        s = [1, S...][i]
        s_ = [S..., 1][i]

        g11 = p[i][1:s * s_, :]
        g21 = p[i][s * s_ + 1:end, :]
        tmp1 = g21 / g11
        tmp2 = tmp1' * tmp1
        Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'

        tmp = (I - Gamma12 * g21) / g11
        summ = summ + tr((Y[i]' * X[i]) * (tmp * tmp' + Gamma12 * Gamma12'))
    end

    return summ
end#=}}}=#

@doc raw"""
    TODO
"""
function project(M::TTTensor{𝔽,V,S}, p, A) where {𝔽,V,S}
    # TODO: project rank 1 tensor A to T_p M
    # Implement this using minres or gmres from IterativeSolvers.jl
end

@doc raw"""
    zero_vector(M::TTTensor{𝔽,V,S}, p)
"""
function zero_vector(M::TTTensor{𝔽,V,S}, p) where {𝔽,V,S}#={{{=#

    return [zeros(n, S) for n in V]
end#=}}}=#

manifold_dimension(::TTTensor{𝔽,V,S}) where {𝔽,V,S} = 0 # TODO

@doc raw"""
    rand(M::TTTensor{ℝ,V,S}; vector_at=nothing)
"""
function rand(M::TTTensor{ℝ,V,S}; vector_at=nothing, kwargs...) where {V,S}#={{{=#
    if isnothing(vector_at)
        return [vcat(rand(GeneralLinear(s * s_)), rand(n - s * s_, s * s_)) for (n, s, s_) in zip(V, [1, S...], [S..., 1])]
    else
        # X = similar.(vector_at) # Initialize
        X = [rand(s1, s2) for (s1, s2) in size.(vector_at)] # Initialize
        Ys = [] # Initialize

        # Get ONB for vertical space
        for (i, s) in enumerate(S); for a in 1:s; for b in 1:s
            Y = similar.(X) # Initialize

            Ks = [zeros(s, s) for s in S]
            Ks[i][a, b] = 1.0

            for (i_, n) in enumerate(V)
                s = [1, S...][i_]
                s_ = [S..., 1][i_]
                K = [Matrix(0I, 1, 1), Ks...][i_]
                K_ = [Ks..., Matrix(0I, 1, 1)][i_]

                # Vertical vector
                g11 = vector_at[i_][1:s * s_, :]
                g21 = vector_at[i_][s * s_ + 1:end, :]
                tmp1 = g21 / g11
                tmp2 = tmp1' * tmp1
                Gamma12 = g11 \ (I - tmp2 / (I + tmp2)) * tmp1'
                Y[i_] .= vector_at[i_] * 
                    (-kronecker(K', Matrix(I, s_, s_)) + kronecker(Matrix(I, s, s), K_)) *
                    ((I - Gamma12 * g21) / (g11' * g11))'
            end

            # Gram-Schmidt
            for Y_ in Ys
                Y .= Y - dot(Y, Y_) * Y_ / dot(Y_, Y_)
            end
            Ys = [Ys..., Y]

        end; end; end

        # Project away from the vertical space
        for Y in Ys
            X .= X - dot(X, Y) * Y / dot(Y, Y)
        end
        
        return X
    end
end#=}}}=#


function Base.show(io::IO, ::TTTensor{𝔽,V,S}) where {𝔽,V,S}
    return print(io, "TTTensor(($(join(V, ", "))), $(S); field=$(𝔽))")
end
