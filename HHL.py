import numpy as np
from qiskit import QuantumCircuit, execute, Aer

x_gate = np.array([[0,1],[1,0]])
z_gate = np.array([[1,0],[0,-1]])
vector = np.array([1,0])

def matrix_to_gate(matrix):
    gate = QuantumCircuit(2)
    gate.unitary(matrix, [0, 1], label='matrix')
    return gate

def solve(counts, num_qubits):
    solution_bits = next(iter(counts.keys()))
    solution_bits = solution_bits[::-1]  
    solution_bits = solution_bits[0:num_qubits-1]  
    solution_decimal = int(solution_bits, 2)
    max_value = 2 ** (num_qubits - 1) - 1
    solution = solution_decimal / max_value

    return solution

def hhl(matrix ,vector):
    num_qubits = int(np.log2(len(vector))) + 1
    qc = QuantumCircuit(num_qubits)
    #prep states
    qc.h(range(num_qubits-1))
    qc.x(num_qubits-1)
    #apply gate
    matrix_gate = matrix_to_gate(matrix)
    qc.append(matrix_gate, range(num_qubits))
    #prep for/and measure
    qc.h(range(num_qubits-1))
    qc.measure_all()

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1).result()
    counts = result.get_counts(qc)
    x_solution = solve(counts, num_qubits)
    y_solution = float(vector[0] - matrix[0][0])

    return [x_solution, y_solution]
    
test1 = hhl(x_gate, vector)
print(f"X is {test1[0]}, Y is {test1[1]}")
test2 = hhl(z_gate, vector)
print(f"X is {test2[0]}, Y is {test2[1]}")



'''
Example : x + 2y = 3 / x + y = 4
x + 2y becomes [1, 2], x + y becomes [1,1] variables put into matrix
solutions are put into vector [3, 4]

in other form:
ax + by = c / dx + ey = f
matrix = [[a, b], [d, e]]
vector = [c, f]
'''


