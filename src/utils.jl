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


@inline get_mat(A::Function, x, u, p, t) = A(x,u,p,t)
@inline get_mat(A::Union{AbstractMatrix, Number},x,u,p,t) = A