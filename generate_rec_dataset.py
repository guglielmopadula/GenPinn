import meshio
import numpy as np


triangles=meshio.read("data/bunny_coarse_train_0.ply").cells_dict["triangle"]
all_points_rec=np.load("all_points_coarse_train_rec.npy")
for i in range(600):
    points=all_points_rec[i]
    meshio.write_points_cells("data/bunny_coarse_rec_"+str(i)+".ply",points,{"triangle":triangles})
