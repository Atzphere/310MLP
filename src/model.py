import keras
import tensorflow as tf


class MLP(keras.Model):
    def __init__(self, n_inputs, layers, output_func):
        super().__init__()

        # subnetwork used to evaluate atomic potential contributions, evaluated for each atom in a structure.
        self.subnet = keras.Sequential(layers=[
            keras.Input(shape=(n_inputs,))]  # input layer takes in n_inputs number of symmetry function features
            + layers  # hidden layers
            + [keras.layers.Dense(1, activation=output_func)])  # output layer returns individual energy contributions

        self.num_features = n_inputs

    def call(self, inputs, training=False):
        '''
        feed-forward method for the model
        should have the signature we ultimately want the model to have,
        i.e. for one structure: Tensor[StructureFeatures] -> Energy_total

        which then for multiple structures:
            Tensor[Tensor[StructureFeatures]] -> Tensor[Energy_total]

        wherein in reality the outermost Tensor is just a list.

        inputs: shape (num_atoms, num_features) tensor
        '''

        #  subnet.call(Tensor[StructureFeatures]) -> Tensor[EnergyContributions]

        #  we call the subnet on each atom in a structure individually.
        #  first, split the input tensor into individual structures:

        structure_energies = []

        for struct in inputs:
            atom_contributions = self.subnet(struct, training=training)
            structure_energies.append(tf.reduce_sum(atom_contributions))

        return structure_energies
        # for structure in inputs:
        #     atom_contributions = self.subnet(structure, training=training)
        #     structure_energy = tf.reduce_sum(atom_contributions)
        #     structure_energies.append(structure_energy)

        # return structure_energies
