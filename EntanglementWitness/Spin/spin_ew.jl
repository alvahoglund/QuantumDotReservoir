using PyCall
PyCall.python
np = pyimport("numpy")

nbr_dots_main = 2
nbr_dots_reservoir = 3
qn_reservoir = 1
qd_system = tight_binding_system(nbr_dots_main,nbr_dots_reservoir, qn_reservoir)

measurements_train, labels_train = generate_dataset(qd_system, 10000, 10000, singlet)
measurements_test, labels_test = generate_dataset(qd_system, 10000, 10000, singlet)

np.save("EntanglementWitness/Spin/Data/measurements_train.npy", measurements_train)
np.save("EntanglementWitness/Spin/Data/labels_train.npy", labels_train)
np.save("EntanglementWitness/Spin/Data/measurements_test.npy", measurements_test)
np.save("EntanglementWitness/Spin/Data/labels_test.npy", labels_test)

