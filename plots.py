import numpy as np
import matplotlib.pyplot as plt
err_ffd=np.load("pinn_simulations/err_fem_ffd.npy")
err_nf=np.load("pinn_simulations/err_fem_nf.npy")
err_ebm=np.load("pinn_simulations/err_fem_ebm.npy")
err_dd=np.load("pinn_simulations/err_fem_dd.npy")

all_err=np.concatenate((err_nf.reshape(1,-1), err_ebm.reshape(1,-1), err_dd.reshape(1,-1), err_ffd.reshape(1,-1)), axis=0)

print(err_ffd)
print(err_nf)
print(err_ebm)
print(err_dd)


plt.semilogy(all_err[:,0], markersize=10, color='r', marker='p', linestyle='none')
plt.suptitle("L1 error between FEM and PINN")
plt.grid(which='major')
plt.grid(which='minor', linestyle='dotted')
plt.gca().set_xticks(np.arange(4),["NF","EBM","DDPM","FFD"])

plt.savefig("L1_error.pdf")

plt.clf()

plt.semilogy(all_err[:,1], markersize=10, color='r', marker='p', linestyle='none')
plt.suptitle("L2 error between FEM and PINN")
plt.grid(which='major')
plt.grid(which='minor', linestyle='dotted')
plt.gca().set_xticks(np.arange(4),["NF","EBM","DDPM","FFD"])

plt.savefig("L2_error.pdf")

plt.clf()