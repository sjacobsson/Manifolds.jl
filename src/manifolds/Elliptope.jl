@doc raw"""
    Elliptope{T} <: AbstractDecoratorManifold{ℝ}

The Elliptope manifold, also known as the set of correlation matrices, consists of all
symmetric positive semidefinite matrices of rank ``k`` with unit diagonal, i.e.,

````math
\begin{aligned}
\mathcal E(n,k) =
\bigl\{p ∈ ℝ^{n×n}\ \big|\ &a^\mathrm{T}pa \geq 0 \text{ for all } a ∈ ℝ^{n},\\
&p_{ii} = 1 \text{ for all } i=1,\ldots,n,\\
&\text{and } p = qq^{\mathrm{T}} \text{ for } q \in  ℝ^{n×k} \text{ with } \operatorname{rank}(p) = \operatorname{rank}(q) = k
\bigr\}.
\end{aligned}
````

And this manifold is working solely on the matrices ``q``. Note that this ``q`` is not unique,
indeed for any orthogonal matrix ``A`` we have ``(qA)(qA)^{\mathrm{T}} = qq^{\mathrm{T}} = p``,
so the manifold implemented here is the quotient manifold. The unit diagonal translates to
unit norm columns of ``q``.

The tangent space at ``p``, denoted ``T_p\mathcal E(n,k)``, is also represented by matrices
``Y\in ℝ^{n×k}`` and reads as

````math
T_p\mathcal E(n,k) = \bigl\{
X ∈ ℝ^{n×n}\,|\,X = qY^{\mathrm{T}} + Yq^{\mathrm{T}} \text{ with } X_{ii} = 0 \text{ for } i=1,\ldots,n
\bigr\}
````
endowed with the [`Euclidean`](@ref) metric from the embedding, i.e. from the ``ℝ^{n×k}``


This manifold was for example
investigated in[JourneeBachAbsilSepulchre:2010](@cite).

# Constructor

    Elliptope(n::Int, k::Int; parameter::Symbol=:type)

generates the manifold ``\mathcal E(n,k) \subset ℝ^{n×n}``.

`parameter`: whether a type parameter should be used to store `n` and `k`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct Elliptope{T} <: AbstractDecoratorManifold{ℝ}
    size::T
end

active_traits(f, ::Elliptope, args...) = merge_traits(IsEmbeddedManifold())

function Elliptope(n::Int, k::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n, k))
    return Elliptope{typeof(size)}(size)
end

@doc raw"""
    check_point(M::Elliptope, q; kwargs...)

checks, whether `q` is a valid representation of a point ``p=qq^{\mathrm{T}}`` on the
[`Elliptope`](@ref) `M`, i.e. is a matrix
of size `(N,K)`, such that ``p`` is symmetric positive semidefinite and has unit trace.
Since by construction ``p`` is symmetric, this is not explicitly checked.
Since ``p`` is by construction positive semidefinite, this is not checked.
The tolerances for positive semidefiniteness and unit trace can be set using the `kwargs...`.
"""
function check_point(M::Elliptope, q; kwargs...)
    row_norms_sq = sum(abs2, q; dims=2)
    if !all(isapprox.(row_norms_sq, 1.0; kwargs...))
        return DomainError(
            row_norms_sq,
            "The point $(q) does not represent a point p=qq^T on $(M) diagonal is not only ones.",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::Elliptope, q, Y; kwargs... )

Check whether ``X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}`` is a tangent vector to
``p=qq^{\mathrm{T}}`` on the [`Elliptope`](@ref) `M`,
i.e. `Y` has to be of same dimension as `q` and a ``X`` has to be a symmetric matrix with
zero diagonal.

The tolerance for the base point check and zero diagonal can be set using the `kwargs...`.
Note that symmetric of ``X`` holds by construction an is not explicitly checked.
"""
function check_vector(
    M::Elliptope,
    q,
    Y::T;
    atol::Real=sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
    kwargs...,
) where {T}
    X = q * Y' + Y * q'
    n = diag(X)
    if !all(isapprox.(n, 0.0; atol=atol, kwargs...))
        return DomainError(
            n,
            "The vector $(X) is not a tangent to a point on $(M) (represented py $(q) and $(Y), since its diagonal is nonzero.",
        )
    end
    return nothing
end

function get_embedding(::Elliptope{TypeParameter{Tuple{n,k}}}) where {n,k}
    return Euclidean(n, k)
end
function get_embedding(M::Elliptope{Tuple{Int,Int}})
    n, k = get_parameter(M.size)
    return Euclidean(n, k; parameter=:field)
end

"""
    is_flat(::Elliptope)

Return false. [`Elliptope`](@ref) is not a flat manifold.
"""
is_flat(M::Elliptope) = false

@doc raw"""
    manifold_dimension(M::Elliptope)

returns the dimension of
[`Elliptope`](@ref) `M` ``=\mathcal E(n,k), n,k ∈ ℕ``, i.e.
````math
\dim \mathcal E(n,k) = n(k-1) - \frac{k(k-1)}{2}.
````
"""
function manifold_dimension(M::Elliptope)
    N, K = get_parameter(M.size)
    return N * (K - 1) - div(K * (K - 1), 2)
end

"""
    project(M::Elliptope, q)

project `q` onto the manifold [`Elliptope`](@ref) `M`, by normalizing the rows of `q`.
"""
project(::Elliptope, ::Any)

project!(::Elliptope, r, q) = copyto!(r, q ./ (sqrt.(sum(abs2, q, dims=2))))
project(::Elliptope, q) = q ./ (sqrt.(sum(abs2, q, dims=2)))

"""
    project(M::Elliptope, q, Y)

Project `Y` onto the tangent space at `q`, i.e. row-wise onto the oblique manifold.
"""
project(::Elliptope, ::Any...)

function project!(::Elliptope, Z, q, Y)
    Y2 = (Y' - q' .* sum(q' .* Y', dims=1))'
    Z .= Y2 - q * lyap(q' * q, q' * Y2 - Y2' * q)
    return Z
end

@doc raw"""
    retract(M::Elliptope, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting ``q+Y`` back onto the manifold.
"""
retract(::Elliptope, ::Any, ::Any, ::ProjectionRetraction)

function retract_project!(M::Elliptope, r, q, Y)
    return ManifoldsBase.retract_project_fused!(M, r, q, Y, one(eltype(q)))
end

function ManifoldsBase.retract_project_fused!(M::Elliptope, r, q, Y, t::Number)
    r .= q .+ t .* Y
    project!(M, r, r)
    return r
end

@doc raw"""
    representation_size(M::Elliptope)

Return the size of an array representing an element on the
[`Elliptope`](@ref) manifold `M`, i.e. ``n×k``, the size of such factor of ``p=qq^{\mathrm{T}}``
on ``\mathcal M = \mathcal E(n,k)``.
"""
representation_size(M::Elliptope) = get_parameter(M.size)

function Base.show(io::IO, ::Elliptope{TypeParameter{Tuple{n,k}}}) where {n,k}
    return print(io, "Elliptope($(n), $(k))")
end
function Base.show(io::IO, M::Elliptope{Tuple{Int,Int}})
    n, k = get_parameter(M.size)
    return print(io, "Elliptope($(n), $(k); parameter=:field)")
end

"""
    vector_transport_to(M::Elliptope, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(::Elliptope, ::Any, ::Any, ::Any, ::ProjectionTransport)
vector_transport_to_project!(M::Elliptope, Y, p, X, q) = project!(M, Y, q, X)

@doc raw"""
    zero_vector(M::Elliptope,p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`Elliptope`](@ref) manifold `M`.
"""
zero_vector(::Elliptope, ::Any...)

zero_vector!(::Elliptope, X, ::Any) = fill!(X, 0)
