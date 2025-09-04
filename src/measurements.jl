sx(i, f) = (f[i, :↓]'*f[i, :↑] + f[i, :↑]'*f[i, :↓])
sy(i, f) = im*(f[i, :↓]'*f[i, :↑] - f[i, :↑]'*f[i, :↓])
sz(i, f) = (f[i, :↑]'*f[i, :↑] - f[i, :↓]'*f[i, :↓])

nbr_op(n :: Integer, f) = f[n, :↑]'*f[n, :↑] + f[n, :↓]'*f[n, :↓]
nbr2_op(n :: Integer, f) = f[n, :↑]'*f[n, :↑]*f[n, :↓]'*f[n, :↓]

process_complex(value, tolerance=1e-3) = abs(imag(value)) < tolerance ? real(value) : throw(ArgumentError("The value has an imaginary part: $(imag(value))"))
expectation_value(ρ, op) = process_complex((tr(ρ * op)))