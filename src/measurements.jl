## ============ Measurements =================
σ0(i, f) = (f[i, :↑]'*f[i, :↑] + f[i, :↓]'*f[i, :↓])
σx(i, f) = (f[i, :↓]'*f[i, :↑] + f[i, :↑]'*f[i, :↓])
σy(i, f) = im*(f[i, :↓]'*f[i, :↑] - f[i, :↑]'*f[i, :↓])
σz(i, f) = (f[i, :↑]'*f[i, :↑] - f[i, :↓]'*f[i, :↓])

nbr_op(coordinate, f) = f[coordinate, :↑]'*f[coordinate, :↑] + f[coordinate, :↓]'*f[coordinate, :↓]
nbr2_op(coordinate, f) = f[coordinate, :↑]'*f[coordinate, :↑]*f[coordinate, :↓]'*f[coordinate, :↓]

pauli_string(σi, σj, qd_system) = tensor_product((matrix_representation(σi(qd_system.coordinates_main[1], qd_system.f), qd_system.H_main_a), 
                matrix_representation(σj(qd_system.coordinates_main[2], qd_system.f), qd_system.H_main_b)),
                (qd_system.H_main_a, qd_system.H_main_b) => qd_system.H_main)


process_complex(value, tolerance=1e-3) = abs(imag(value)) < tolerance ? real(value) : throw(ArgumentError("The value has an imaginary part: $(imag(value))"))
expectation_value(ρ, op) = process_complex((tr(ρ * op)))

## ======== Measurement sets =================

pauli_strings(qd_system) = [pauli_string(σi, σj, qd_system) for σi in [σ0, σx, σy, σz] for σj in [σ0, σx, σy, σz]]

single_charge_measurements(qd_system) = [nbr_op(coordinate, qd_system.f) for coordinate in qd_system.coordinates_total]
double_charge_measurements(qd_system) = [nbr2_op(coordinate, qd_system.f) for coordinate in qd_system.coordinates_total]
charge_measurements(qd_system) = vcat(single_charge_measurements(qd_system), double_charge_measurements(qd_system))