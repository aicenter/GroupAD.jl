function plot_number(bag)
    if maximum(bag) <= 1
        scatter(bag[1,:], bag[2,:], color=:black, legend=:none, markersize=bag[3,:] .* 5, aspect_ratio=:equal, xlims=(0, 1), ylims=(0, 1), size=(400, 400), grid=:none, ticks=:none)
    elseif minimum(bag) < 0
        scatter(
            bag[1,:], bag[2,:], label="",
            markersize=(bag[3,:]) .+ abs(minimum(bag[3,:])),
            aspect_ratio=:equal, size=(400, 400),
            axis=([],false)
        )
    else
        scatter(bag[1,:], bag[2,:], color=:black, markersize=bag[3,:] ./ 40, aspect_ratio=:equal, xlims=(0, 28), ylims=(0, 28), size=(400, 400))
    end
end

normal, anomalous, ln, la = GroupAD.load_data("MNIST",anomaly_class=4, noise=false, normalize=false)
k = 1
p1 = plot_number(normal[k].data.data);
p2 = plot_number(anomalous[k].data.data);
plot(p1,p2,layout=(1,2),size=(600,300))
k += 1

"""
Někde je chyba při převodu dat, protože dostáváme některé číslice, které jsou 
jakoby kombinací jiných, třeba dohromady 3 a 9, což by samozřejmě nadělalo docela paseku.
"""

function plot_number_row(start_index,data,labels)
    p = plot(layout=(5,5),axis=([],false),size=(700,700))
    for i in 1:25
        bag = data[start_index+i-1]
        p = scatter!(p,
            bag[1,:], bag[2,:],label="$(labels[start_index+i-1])",
            legend=:outertop,
            markersize=(bag[3,:])./70,
            aspect_ratio=:equal,
            axis=([],false),
            subplot=i
        )
    end
    return p
end

data = [normal[i].data.data for i in 1:length(normal)]
plot_number_row(1,data,ln)