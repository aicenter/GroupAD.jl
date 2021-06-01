function plot_roc_scores(labels, scores)
    p = plot()
    for score in scores
        p = rocplot!(labels, score)
    end
    p
end

function plot_pr_scores(labels, scores)
    p = plot()
    for score in scores
        p = prplot!(labels, score)
    end
    p
end

function plot_layout(data,layout;kwargs...)
    l = prod(layout)
    if l != length(data)
        error("Layout size not compatible with data size!")
    end
    p = plot(layout=layout;kwargs...)
    for i in 1:l
        x = data[i]
        if size(x,1) > size(x,2)
            x = x'
        end
        p = scatter!(p, x[1,:],x[2,:], subplot=i)
    end
    return p
end


function scatter2(X; kwargs...)
    if size(X,1) != 2
        error("Input vector needs to habe 2 rows!")
    end
    Plots.scatter(X[1,:],X[2,:]; kwargs...)
end
function scatter2!(X; kwargs...)
    if size(X,1) != 2
        error("Input vector needs to habe 2 rows!")
    end
    Plots.scatter!(X[1,:],X[2,:]; kwargs...)
end