"""
Plotting functions for numbers.
"""

"""
    plot_number(bag)

Plots a bag as a point cloud.
"""
function plot_number(bag)
    if maximum(bag) <= 1
        scatter(
            bag[1,:], bag[2,:], color=:black,
            legend=:none, markersize=bag[3,:] .* 5 .+ 2, aspect_ratio=:equal,
            xlims=(-0.1, 1.1), ylims=(-0.1, 1.1), size=(400, 400), axis=([],false)
        )
    elseif minimum(bag) < 0
        scatter(
            bag[1,:], bag[2,:], label="",
            markersize = 1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            aspect_ratio=:equal, size=(400, 400),
            axis=([],false)
        )
    else
        scatter(bag[1,:], bag[2,:], color=:black, markersize=bag[3,:] ./ 40, aspect_ratio=:equal, xlims=(0, 28), ylims=(0, 28), size=(400, 400))
    end
end

"""
    plot_old_new(bag,model)
Plots original and reconstructed bag side to side.
"""
function plot_old_new(bag,model)
    new_bag = reconstruct(model,bag);
    o_plot = plot_number(bag);
    n_plot = plot_number(new_bag);
    plot(layout=(1,2),o_plot,n_plot,size=(600,300))
end
function plot_old_new2(bag,model)
    new_bag = reconstruct2(model,bag);
    o_plot = plot_number(bag);
    n_plot = plot_number(new_bag);
    plot(layout=(1,2),o_plot,n_plot,size=(600,300))
end

"""
    plot_number_col(start_index,data,model)
Plots a col of original and reconstructed numbers.
"""
function plot_number_col(start_index,data,model)
    p = plot(layout=(5,2),axis=([],false),size=(200,500))
    for i in 1:5
        bag = data[start_index+i-1]
        p = scatter!(p,
            bag[1,:], bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            aspect_ratio=:equal,
            axis=([],false),
            subplot=2i-1
        )
        new_bag = reconstruct(model,bag)
        p = scatter!(p,
            new_bag[1,:], new_bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            color=:green,
            aspect_ratio=:equal,
            axis=([],false),
            subplot=2i
        )
    end
    return p
end

"""
    plot_number_row(start_index,data,model)
Plots a row of original and reconstructed numbers.
"""
function plot_number_row(start_index,data;k=5)
    p = plot(layout=(1,5),axis=([],false),size=(k*100,200))
    for i in 1:k
        bag = data[start_index+i-1]
        p = scatter!(p,
            bag[1,:], bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            aspect_ratio=:equal,
            axis=([],false),
            subplot=i
        )
    end
    return p
end
function plot_number_row(start_index,data,model; color=:auto, k=6)
    p = plot(layout=(2,k),axis=([],false),size=(k*100,200))
    for i in 1:k
        bag = data[start_index+i-1]
        p = scatter!(p,
            color=color,
            bag[1,:], bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            aspect_ratio=:equal,
            axis=([],false),
            subplot=i
        )
        new_bag = reconstruct(model,bag)
        p = scatter!(p,
            new_bag[1,:], new_bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            color=:green,
            aspect_ratio=:equal,
            axis=([],false),
            subplot=i+k
        )
    end
    return p
end

function plot_number_col(start_index,data)
    p = plot(layout=(5,1),axis=([],false),size=(200,500))
    for i in 1:5
        bag = data[start_index+i-1]
        p = scatter!(p,
            bag[1,:], bag[2,:], label="",
            markersize=1.2 .* (bag[3,:] .+ abs(minimum(bag[3,:]))),
            aspect_ratio=:equal,
            axis=([],false),
            subplot=i
        )
    end
    return p
end




"""
    plot_numbers(start_index,data [, model])
Plots 3 columns of original and reconstructed numbers.

Examples:

```
plot_number_col(48,train_data_bag)
plot_number_col(1,test_data_bag[end-10:end])
plot_numbers(48,train_data_bag)
plot_numbers(1,test_data_bag[end-20:end])
```
"""
function plot_numbers(start_index,data)
    p1 = plot_number_col(start_index,data)
    p2 = plot_number_col(start_index+5,data)
    p3 = plot_number_col(start_index+10,data)
    p4 = plot_number_col(start_index+15,data)
    p5 = plot_number_col(start_index+20,data)
    plot(p1,p2,p3,p4,p5,layout=(1,5),size=(600,600))
end
function plot_numbers(start_index,data,model)
    p1 = plot_number_col(start_index,data,model)
    p2 = plot_number_col(start_index+5,data,model)
    p3 = plot_number_col(start_index+10,data,model)
    plot(p1,p2,p3,layout=(1,3),size=(600,500))
end


function plot_na(data, model; nidx = 10, aidx = nothing, k = 10, an_color=2, layout = (2,1))
    # if anomaly start index not given
    isnothing(aidx) ? aidx = length(data) - 50 : nothing

    p1 = plot_number_row(nidx, data, model; k=k)
    p2 = plot_number_row(aidx, data, model; k=k, color=an_color)

    if layout[1] > layout[2]
        return plot(p1,p2,layout=layout,size=(k*100,400))
    else
        return plot(p1,p2,layout=layout,size=(k*2*100,200))
    end
end