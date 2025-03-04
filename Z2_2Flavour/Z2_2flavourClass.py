from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister,ClassicalRegister
from qiskit_machine_learning.neural_networks import EstimatorQNN,SamplerQNN
from IPython.display import clear_output
import matplotlib.pyplot as plt
import qiskit.quantum_info as qi
from qiskit.circuit import ParameterVector
from qiskit_algorithms.utils import algorithm_globals
import time 
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2,RealAmplitudes
import numpy as np

import pickle 
import time
from qiskit.circuit.library import iSwapGate
from math import comb

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
sqrt_iSWAP = iSwapGate().power(1/2)

def getChargeArr(strings,nLinks): #input is a job result
    p_list = []
    
    charge1 = np.zeros(nLinks+1)
    charge2 = np.zeros(nLinks+1)
    weights_all = 0
    
    for key in strings.keys():
        tmpKey = np.array([int(numeric_string) for numeric_string in key])  
        tmpKey = 1-tmpKey
        weight = strings[key]
        #arr = 1 - 2*tmpKey
        
        tmp_charge1 = []
        tmp_charge2 = []
        for i in range(len(tmpKey)):
            if (i+1)%3 == 1:
                tmp_charge1.append(tmpKey[i])
            elif (i+1)%3 == 2:
                tmp_charge2.append(tmpKey[i])
                
        tmp_charge1 = np.array(tmp_charge1)
        tmp_charge2 = np.array(tmp_charge2)

        if tmp_charge1.sum() == 1 and tmp_charge2.sum() == 1:
            weights_all += weight
            charge1 += np.array(tmp_charge1)*weight
            charge2 += np.array(tmp_charge2)*weight
            

    return charge1, charge2, weights_all
            
    
def getChargeLoc_fromString(string_list_dict,nLinks):

    prob1 = []
    prob2 = []
    for key in string_list_dict.keys(): #each time step
        print(key)
        evs_list_tmp = []
        l = string_list_dict[key]
        
        charge1 = np.zeros(nLinks+1)
        charge2 = np.zeros(nLinks+1)
        weights_all = 0
        for strings in l: #bundle of jobs in each time step, iterate to get each job
            tmp_c1, tmp_c2, weight = getChargeArr(strings,nLinks)
            weights_all += weight
            charge1 += np.array(tmp_c1)
            charge2 += np.array(tmp_c2)
            
        charge1 = charge1 / weights_all
        charge2 = charge2 / weights_all

        
        prob1.append(charge1)
        prob2.append(charge2)
        
    return np.array(prob1), np.array(prob2)   


class Z2_2flavour:
    def __init__(self,nLinks = 5, matter_defect=[4,12]):
        '''
        matter_defect: positions of the charges
        '''

        self.nLinks = nLinks
        self.n_qubits = 3*nLinks + 2
        self.matter_defect = matter_defect
        self.circuit = QuantumCircuit(self.n_qubits)

        #First produce the initial state:
        for i in range(self.n_qubits):
            defect_flag=False

            for index in matter_defect:
                if i== index :
                    defect_flag=True

            if defect_flag==False:
                if (i+1)%3!=0:
                    self.circuit.x(i)

                else:
                    self.circuit.h(i)
        
    def add3Q(self,tmpIndex,tmpIndex1,tmpIndex2,alpha = 1):

        self.circuit.append(sqrt_iSWAP, [tmpIndex, tmpIndex1])
        self.circuit.append(sqrt_iSWAP, [tmpIndex, tmpIndex1])
        #circuit.barrier()
        self.circuit.rz(np.pi,tmpIndex)
        self.circuit.rz(-np.pi/4,tmpIndex1)
        self.circuit.rz(-np.pi/4,tmpIndex2)
        self.circuit.append(sqrt_iSWAP, [tmpIndex1, tmpIndex2])

        self.circuit.rz(np.pi-alpha,tmpIndex1)
        self.circuit.rz(alpha,tmpIndex2)
        self.circuit.append(sqrt_iSWAP, [tmpIndex1, tmpIndex2])
        self.circuit.rz(np.pi/4,tmpIndex1)
        self.circuit.rz(np.pi/4,tmpIndex2)
        #circuit.barrier()

        self.circuit.append(sqrt_iSWAP, [tmpIndex, tmpIndex1])
        self.circuit.append(sqrt_iSWAP, [tmpIndex, tmpIndex1])
        
        
    def getTrotterCircuit(self,J,nTrotter=2,fFactor=0.2,dtFactor=0.2,measure=True,rescale=False):
        '''
        No parity consideration yet.
        rescale: whether the circuit is made for depolarization noise measurement.
        measure: Whether the qubits are measured at the end.
        '''
        f=fFactor*J
        dt_mag=dtFactor/J        

        #Trotterization:
        for step in range(nTrotter):
            if rescale:

                if step < int(nTrotter/2):

                    dt = dt_mag
                else: 
                    dt = -dt_mag
            else:
                dt = dt_mag
                
            self.circuit.barrier()
            
            
            for i in range(1, self.nLinks+1, 1):
                if i%2 == 1:
                    tmp = i*3 - 1
                    
                    s1up = tmp - 2
                    s1down = tmp + 1

                    s2up = tmp - 1
                    s2down = tmp + 2

                    self.add3Q(s1up,tmp,s1down,alpha=J*dt)
                    self.add3Q(s2up,tmp,s2down,alpha=J*dt)
                    
            self.circuit.barrier()
                    
            for i in range(1, self.nLinks+1, 1):
                if i%2 == 0:
                    tmp = i*3 - 1
                    
                    s1up = tmp - 2
                    s1down = tmp + 1

                    s2up = tmp - 1
                    s2down = tmp + 2
                    self.add3Q(s1up,tmp,s1down,alpha=J*dt)
                    self.add3Q(s2up,tmp,s2down,alpha=J*dt)
                    
            self.circuit.barrier()


            for i in range(self.n_qubits):
                if (i+1)%3 == 0: #picking links
                    self.circuit.rx(f*dt, i)
                else:
                    self.circuit.rz(0,i) #no mass term.
            self.circuit.barrier()
            
    
        for i in range(self.n_qubits):
            if (i+1)%3 == 0:
                self.circuit.h(i)

        if measure:
            self.circuit.measure_all()
    