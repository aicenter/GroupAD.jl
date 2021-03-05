using Test
using GroupAD

@testset "data" begin
	# MNIST point cloud
	data1 = GroupAD.load_mnist_point_cloud(anomaly_class=1)
	data5 = GroupAD.load_mnist_point_cloud(anomaly_class=5)
	@test size(data1[1],1) == size(data5[1],1)

	# test of reindexing
	N = 1000
	d = data1[1];
	inds = rand(1:length(d), N);
	bg1=d[inds]
	bg2=GroupAD.reindex(d, inds)
	@test all(bg1.data.data .== bg2.data.data)
	@test all(bg1.bags .== bg2.bags)

	# load MNIST point cloud
	(tr_data, tr_y), (val_data, val_y), (tst_data, tst_y) = GroupAD.load_data("MNIST"; anomaly_class=4)
	@test sum(tr_y) == 0
	@test sum(val_y) == sum(tst_y) > 0
end
