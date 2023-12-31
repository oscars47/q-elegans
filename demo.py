# simple file to demonstrate the basic idea of the project
# NOTE: unfortunately, tensorflow quantum can't be installed through pip. Downloaded source code and installed using python setup.py install, but unable to import tensorflow_quantum even though I am in the correct virtual environment. Will try to fix this later.

import cirq
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_quantum as tfq

def build_circuit(classical_input, test=False):
    '''Implements quantum circuit to be used in the tfq model.

    Params:
        num_inputs: number of qubits in the circuit
        test: if true, will print the circuit and state vector, simulated 1000 times
    
    Returns:
        cirq circuit
    '''
    assert np.isclose(np.linalg.norm(classical_input), 1, atol = 1e-10), f'classical data must be normalized. Norm is {np.linalg.norm(classical_input)}'
    
    num_inputs = len(classical_input)
    # create a num_inputs qubit system
    qubits = [cirq.GridQubit(i, 0) for i in range(num_inputs)]  # create qubits

    # create a circuit
    circuit = cirq.Circuit(cirq.H(q) for q in qubits)  # Apply a Hadamard gate to each qubit

    # encode classical data
    for i, data in enumerate(classical_input):
        # Map data to angle
        phi = np.pi * data 
        circuit.append(cirq.ry(phi)(qubits[i]))

    # entangle all qubits with CNOT; raise to power of theta_ij to implement partial entanglement
    k = 0
    for i in range(len(qubits) - 1):
        for j in range(i + 1, len(qubits)):
            circuit.append(cirq.CNOT(qubits[i], qubits[j])**sp.Symbol('theta_' + str(i) + '_' + str(j)))
            k+=1

    print('k = ', k)
    print(len(qubits) * (len(qubits) - 1) // 2)
    print([i + j for i in range(len(qubits)-1) for j in range(i + 1, len(qubits))])

    # entangle only adjacent
    # for i in range(len(qubits) - 1):
    #     circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

    # rotate qubits in X, Y, Z
    for i in range(len(qubits)):
        circuit.append(cirq.rx(sp.Symbol('alpha_' + str(i)))(qubits[i]))
        circuit.append(cirq.ry(sp.Symbol('beta_' + str(i)))(qubits[i]))
        circuit.append(cirq.rz(sp.Symbol('gamma_' + str(i)))(qubits[i]))

    if test:
        # set the values of theta, alpha, beta, gamma
        thetas = np.random.rand(len(qubits) * (len(qubits) - 1) // 2)
        alphas = np.random.rand(len(qubits))
        betas = np.random.rand(len(qubits))
        gammas = np.random.rand(len(qubits))

        # set the values of the symbols in the circuit
        param_resolver = {'theta_' + str(i) + '_' + str(j): thetas[i + j] for i in range(len(qubits)-1) for j in range(i + 1, len(qubits))}
        param_resolver.update({'alpha_' + str(i): alphas[i] for i in range(len(alphas))})
        param_resolver.update({'beta_' + str(i): betas[i] for i in range(len(betas))})
        param_resolver.update({'gamma_' + str(i): gammas[i] for i in range(len(gammas))})
        param_resolver = cirq.ParamResolver(param_resolver)

        # print the circuit
        print("Circuit:")
        print(circuit)
        # measure in the computational basis
        circuit.append(cirq.measure(*qubits, key='z'))
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000, param_resolver=param_resolver)

        # plot the histogram using matplotlib
        hist_counter = result.histogram(key='z')
        plt.bar(hist_counter.keys(), hist_counter.values())
        plt.show()
        
    return qubits, circuit

def build_model(classical_input, output):
    print(tfq.__version__)
    num_inputs = len(classical_input)
    output_len = len(output)
    qubits, circuit = build_circuit(classical_input)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_inputs,), dtype=tf.dtypes.float32),
        tfq.layers.PQC(
            circuit, operators=cirq.Z(qubits)
            ),
        tf.keras.layers.Dense(output_len, activation='sigmoid')
    ])
    # complete model
    model = tf.keras.models.Model(inputs=[classical_input], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['val_accuracy'])
    return model

if __name__ == "__main__":
    # test the circuit
    data = np.random.rand(4)
    data /= np.linalg.norm(data)
    data1 = np.random.rand(4)
    data1 /= np.linalg.norm(data1)
    # build_circuit(data, test=True)
    build_model(data, data1)


    

    # q_layer = tfq.layers.PQC( # create the parameterized quantum circuit as a layer
    #     cirq.resolve_parameters(circuit, {'theta': np.pi / 4}), # convert symbolic param to numeric values
    #     operators=cirq.Z(qubits[1]) # measure qubit 1 in the computational basis
    # )

    # # classical input
    # classical_input = tf.keras.layers.Input(shape=(1,), dtype=tf.dtypes.float32)

    # # quantum part
    # quantum_input = tfq.convert_to_tensor([circuit] * 1)  # Assuming batch size of 1 for simplicity
    # quantum_output = q_layer(quantum_input)

    # # combine classical and quantum parts
    # combined = tf.keras.layers.Concatenate()([classical_input, quantum_output])

    # # final dense layers
    # output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

    

  