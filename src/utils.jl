function sample_discrete(prob, n_particules; n_exp = 1)

    # this speedup is due to Peter Acklam
    cumprob = cumsum(prob, dims=1)

    N = size(cumprob, 1)
    R = rand(n_exp, n_particules)

    ind = ones(Int64, (n_exp, n_particules))
    for i = 1:N-1
        ind .+= R .> cumprob[i, :]
    end
    ind

end


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
    inv_using_SVD(Mat, eigvalMax)

SVD decomposition of Matrix. 
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
        U_3 = @view F.U[icut+1:end, 1:icut]
        V_3 = @view F.Vt'[icut+1:end, 1:icut]
        tmp2 = (V_1 ./ F.S[1:icut]') * U_3'
        tmp3 = (V_3 ./ F.S[1:icut]') * U_1'
        tmp4 = (V_3 ./ F.S[1:icut]') * U_3'
        vcat(hcat(tmp1, tmp2), hcat(tmp3, tmp4))
    end

end


"""
    sqrt_svd(A)  

Returns the square root matrix by SVD
"""
function sqrt_svd(A::AbstractMatrix)
    F = svd(A)
    F.U * diagm(sqrt.(F.S)) * F.Vt
end


@inline get_mat(A::Function, x, u, p, t) = A(x,u,p,t)
@inline get_mat(A::Union{AbstractMatrix, Number}, x, u, p, t) = A

# @inline get_M(sys::GaussianLinearStateSpaceSystem, x, exogenous_variables[t, :], control_variables[t, :], parameters)
@inline scale(x, μ=0.0, σ=1.0) = (x .- μ)./σ
function standard_scaler(X::Array{Float64, 2}; with_mean=true, with_std=true)

    n_X = size(X, 2)
    μ = with_mean ? mean(X, dims=1) : zeros(1, n_X)
    σ = with_std ? std(X, dims=1) : ones(1, n_X)
    return scale(X, μ, σ)

end