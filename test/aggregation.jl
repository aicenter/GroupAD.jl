@testset "agreggation" begin
	(tr_b, tr_y), (val_b, val_y), (tst_b, tst_y) = GroupAD.load_data("Fox")
	tr_x = GroupAD.Models.aggregate(tr_b, mean)
	@test size(tr_x) == (size(tr_b.data.data,1), length(tr_b.bags))
	@test eltype(tr_x) == eltype(tr_b.data.data)
end