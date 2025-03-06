# using FunctionWrappers
# import FunctionWrappers: FunctionWrapper

"""
Discrete sampling over ``n_particules`` with ``prob`` probabilities.

$(TYPEDSIGNATURES)
"""
function sample_discrete(prob, n_particules; n_exp = 1)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob, dims = 1)

    N = size(cumprob, 1)
    R = rand(n_exp, n_particules)

    ind = ones(Int64, (n_exp, n_particules))
    for i in 1:(N - 1)
        ind .+= R .> cumprob[i, :]
    end
    ind
end

"""
Resample indexes of ``indx`` using weights ``w``.

$(TYPEDSIGNATURES)
"""
function resample!(indx::Vector{Int64}, w::Vector{Float64})
    m = length(w)
    q = cumsum(w)
    i = 1
    while i <= m
        sampl = rand()
        j = 1
        while q[j] < sampl
            j = j + 1
        end
        indx[i] = j
        i = i + 1
    end
end

"""
SVD decomposition of Matrix.

$(TYPEDSIGNATURES)
"""
function inv_using_SVD(Mat, eigvalMax)
    F = svd(Mat; full = true)
    eigval = cumsum(F.S) ./ sum(F.S)
    # search the optimal number of eigen values
    icut = findfirst(eigval .>= eigvalMax)

    U_1 = @view F.U[1:icut, 1:icut]
    V_1 = @view F.Vt'[1:icut, 1:icut]
    tmp1 = (V_1 ./ F.S[1:icut]') * U_1'

    if icut + 1 > length(eigval)
        tmp1
    else
        U_3 = @view F.U[(icut + 1):end, 1:icut]
        V_3 = @view F.Vt'[(icut + 1):end, 1:icut]
        tmp2 = (V_1 ./ F.S[1:icut]') * U_3'
        tmp3 = (V_3 ./ F.S[1:icut]') * U_1'
        tmp4 = (V_3 ./ F.S[1:icut]') * U_3'
        vcat(hcat(tmp1, tmp2), hcat(tmp3, tmp4))
    end
end

"""
Returns the square root matrix by SVD

$(TYPEDSIGNATURES)
"""
function sqrt_svd(A::AbstractMatrix)
    F = svd(A)
    F.U * diagm(sqrt.(F.S)) * F.Vt
end

@inline _scale(x, μ = 0.0, σ = 1.0) = (x .- μ) ./ σ
"""
Standard scaling of Matrix ``X``

$(TYPEDSIGNATURES)
"""
function standard_scaler(X::Matrix; with_mean = true, with_std = true)
    n_X = size(X, 2)
    μ = with_mean ? mean(X, dims = 1) : zeros(1, n_X)
    σ = with_std ? std(X, dims = 1) : ones(1, n_X)
    return _scale(X, μ, σ)
end

###########################################################################################################
# MatrixProviders
###########################################################################################################

"""
$(TYPEDEF)

FunctionWrapper for function that takes as input (Vector{T}, Vector{<:Real}, T) and returns a Matrix{T}.
"""
# const LinearMatFunction{T} = FunctionWrapper{
#     Matrix{T}, Tuple{Vector{T}, Vector{<:Real}, T}} where {T <: Real}

"""
$(TYPEDEF)

FunctionWrapper for function that takes as input (Vector{T}, Vector{<:Real}, T) and returns a Vector{T}.
"""
# const LinearVecFunction{T} = FunctionWrapper{
#     Vector{T}, Tuple{Vector{T}, Vector{<:Real}, T}} where {T <: Real}

# TODO : see if it's possible to replace VecOrMat by Matrix or Vector for NonLinearMatorVecFunction?
"""
$(TYPEDEF)

FunctionWrapper for function that takes as input (VecOrMat{T}, Vector{T}, Vector{<:Real}, T) and returns a VecOrMat{T}.
"""
# const NonLinearMatorVecFunction{T} = FunctionWrapper{
#     VecOrMat{T}, Tuple{VecOrMat{T}, Vector{T}, Vector{T}, Vector{<:Real}, T}} where {T <: Real}

"""
$(TYPEDEF)

AbstractProvider is an abstract type defined for elements in AbstractStateSpaceSystem that allows 
to generate transition and observation equations.
"""
abstract type AbstractProvider{Z <: Real} end

"""
$(TYPEDEF)

LinearAbstractProvider is a subtype of AbstractProvider for linear expressions with 
respect to x and u.
"""
abstract type LinearAbstractProvider{Z <: Real} <: AbstractProvider{Z} end

"""
$(TYPEDEF)

AbstractProvider is a subtype of AbstractProvider for subtype returning a Vector{Z}.
"""
abstract type AbstractVectorProvider{Z <: Real} <: LinearAbstractProvider{Z} end
"""
$(TYPEDEF)

AbstractProvider is a subtype of AbstractProvider for subtype returning a Matrix{Z}.
"""
abstract type AbstractMatrixProvider{Z <: Real} <: LinearAbstractProvider{Z} end

"""
$(TYPEDEF)

StaticMatrix is a subtype of AbstractMatrixProvider containing Matrix{Z}.
"""
struct StaticMatrix{Z <: Real, P <: AbstractMatrix{Z}} <: AbstractMatrixProvider{Z}
    value::P

    function StaticMatrix(m::P) where {Z <: Real, P <:AbstractMatrix{Z}}
        return new{Z, P}(m)
    end
end

"""
$(TYPEDEF)

StaticVector is a subtype of AbstractVectorProvider containing Vector{Z}.
"""
struct StaticVector{Z <: Real, P <: AbstractVector{Z}} <: AbstractVectorProvider{Z}
    value::P

    function StaticVector(m::P) where {Z <: Real, P <:AbstractVector{Z}}
        return new{Z, P}(m)
    end
end

"""
$(TYPEDEF)

DynamicMatrix is a subtype of AbstractMatrixProvider containing MatFunction{Z} 
that returns a Matrix{Z}.
"""
struct DynamicMatrix{Z <: Real} <: AbstractMatrixProvider{Z}
    func::Function#LinearMatFunction{Z}

    # function DynamicMatrix{Z}(f::Function) where {Z <: Real}
    #     new{Z}(LinearMatFunction{Z}(f))
    # end
end

"""
$(TYPEDEF)

DynamicVector is a subtype of AbstractVectorProvider containing VecFunction{Z} 
that returns a Vector{Z}.
"""
struct DynamicVector{Z <: Real} <: AbstractVectorProvider{Z}
    func::Function #LinearVecFunction{Z}

    # function DynamicVector{Z}(f::Function) where {Z <: Real}
    #     new{Z}(LinearVecFunction{Z}(f))
    # end
end

"""
$(TYPEDEF)

NonLinearProvider is a subtype of AbstractProvider containing NonLinearMatorVecFunction{Z} 
for provider expressing non linear relations with respect to x and u.
"""
abstract type AbstractNonLinearProvider{Z <: Real} <: AbstractProvider{Z} end

struct TransitionNonLinearProvider{Z <: Real} <: AbstractNonLinearProvider{Z}
    func::Function #NonLinearMatorVecFunction{Z}

    # function TransitionNonLinearProvider{Z}(f::Function) where {Z <: Real}
    #     new{Z}(NonLinearMatorVecFunction{Z}(f))
    # end
end

struct TransitionNonParametricProvider{Z <: Real} <: AbstractNonLinearProvider{Z}
    func::Function #NonLinearMatorVecFunction{Z}

    # function TransitionNonLinearProvider{Z}(f::Function) where {Z <: Real}
    #     new{Z}(NonLinearMatorVecFunction{Z}(f))
    # end
end

struct ObservationNonLinearProvider{Z <: Real} <: AbstractNonLinearProvider{Z}
    func::Function #NonLinearMatorVecFunction{Z}

    # function ObservationNonLinearProvider{Z}(f::Function) where {Z <: Real}
    #     new{Z}(NonLinearMatorVecFunction{Z}(f))
    # end
end


# """
# Define `call` operator for LinearAbstractProvider with exogenous parameters as view.

# $(TYPEDSIGNATURES)
# """
# @inline function (A::LinearAbstractProvider{Z})(
#         exogenous::SubArray{Z}, params, t) where {Z <: Real}
#     return A(exogenous, params, t)
# end

"""
Define `call` operator for StaticMatrix.

$(TYPEDSIGNATURES)
"""
@inline function (A::StaticMatrix{Z, P})(exogenous, params, t)::P where {Z <: Real, P <: AbstractMatrix{Z}}
    return A.value::P
end

"""
Define `call` operator for StaticVector.

$(TYPEDSIGNATURES)
"""
@inline function (A::StaticVector{Z, P})(exogenous, params, t)::P where {Z <: Real, P <: AbstractVector{Z}}
    return A.value::P
end

"""
Define `call` operator for DynamicMatrix.

$(TYPEDSIGNATURES)
"""
@inline function (A::DynamicMatrix)(exogenous, params, t)# where {Z <: Real, D <: Real}
    return A.func(exogenous, params, t)
end

"""
Define `call` operator for DynamicVector.

$(TYPEDSIGNATURES)
"""
@inline function (A::DynamicVector)(exogenous, params, t)# where {Z <: Real, D <: Real}
    return A.func(exogenous, params, t)
end

"""
Define `call` operator for NonLinearProvider with exogenous parameters as view.

$(TYPEDSIGNATURES)
"""
@inline function (A::TransitionNonLinearProvider{Z})(
        x, exogenous::SubArray{Z}, u, params, t) where {Z <: Real}
    return A(x, copy(exogenous), u, params, t)
end

"""
Define `call` operator for NonLinearProvider (Transition Operator).

$(TYPEDSIGNATURES)
"""
@inline function (A::TransitionNonLinearProvider{Z})(x, exogenous, u, params, t) where {Z <: Real}
    return A.func(x, exogenous, u, params, t)
end

"""
Define `call` operator for TransitionNonParametricProvider (Transition Operator).

$(TYPEDSIGNATURES)
"""
@inline function (A::TransitionNonParametricProvider{Z})(x, x_scaled, exogenous, u, params, llrs, t) where {Z <: Real}
    return A.func(x, x_scaled, exogenous, u, params, llrs, t)
end

"""Observation operator."""
@inline function (A::ObservationNonLinearProvider{Z})(x, exogenous, params, t) where {Z <: Real}
    return A.func(x, exogenous, params, t)
end

function _promote(args...)
    _type = Base.promote_eltype(args...)
    return map(x -> convert.(_type, x), args)
end

######################################## Rolling Std ################################################""

mutable struct RollingStd{Z}
    count::Int
    buffer::Vector{Z}
    max_size::Int

    function RollingStd{Z}(max_size::Int) where {Z <: Real}
        new{Z}(0.0, Z[], max_size)
    end
end
@inline RollingStd(max_size::Int) = RollingStd{DEFAULT_REAL_TYPE}(max_size)

function update!(rs::RollingStd{Z}, new_data::Z) where {Z <: Real}
    if rs.count < rs.max_size
        push!(rs.buffer, new_data)
        rs.count += 1
    else
        popfirst!(rs.buffer)
        push!(rs.buffer, new_data)
    end
end

function get_value(rs::RollingStd{Z}) where {Z <: Real}
    if rs.count < rs.max_size
        return NaN
    else
        return std(rs.buffer)
    end
end