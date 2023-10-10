dataset = "modelnet"
method = "chair"

parameters = sample_params()
data = GroupAD.load_data(dataset, method=method)
model = GroupAD.Models.pm_constructor(;idim=size(data[1][1],1), parameters...)

opt = ADAM()

using GroupAD.Models: unpack_mill, RandomBagBatches

tr_x, tr_l = unpack_mill(data[1])
vx, vl = unpack_mill(data[2])
val_x = vx[vl .== 0]

lossf(x) = loss(tr_model, x)

bag_batch = RandomBagBatches(tr_x,batchsize=2,randomize=true)

tr_model = deepcopy(model)
ps = Flux.params(tr_model)

lossf.(bag_batch)
tr_model(bag_batch[1])

@time Flux.train!(lossf, ps, bag_batch, opt)
