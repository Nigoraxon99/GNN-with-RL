import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq)-1):
            if graph.get_edge_data(seq[i], seq[i+1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i+1])['weight']+1
            graph.add_edge(seq[i], seq[i+1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois+item_tail * (len_max-le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le+[0] * (len_max-le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1.-valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, sub_graph=False, sparse=False, shuffle=False):
        inputs = data[0]
        # print(f'item sequence', data)
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        # print(f'self.targets', self.targets)
        self.length = len(inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index, dataset):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            if dataset == 'sample':
                max_n_items = 32
            elif dataset == 'diginetica':
                max_n_items = 82
            else:
                max_n_items = 146
            max_length = 0
            for u_input in self.inputs[index]:
                length = len(u_input)
                if length > max_length:
                    max_length = length
            for u_input in self.inputs[index]:
                node = np.unique(u_input)
                alias_inputs.append(
                    np.array([np.where(node == i)[0][0] for i in u_input]+(max_n_items-max_length) * [0]))

                items.append(node.tolist()+(max_n_items-len(node)) * [0])
                u_A = np.zeros((max_n_items, max_n_items))
                for i in np.arange(max_n_node):
                    if u_input[i+1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i+1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)

                A_in.append(u_A_in)
                A_out.append(u_A_out)
            mask = self.mask[index]
            # print(f'mask shape', self.mask.shape[1])
            padded_mask = np.pad(mask, ((0, 0), (0, max_n_items-self.mask.shape[1])), mode='constant',
                                 constant_values=0)

            return A_in, A_out, alias_inputs, items, padded_mask, self.targets[index], max_n_node

    def get_updated_slice(self, index, recommended_items, dataset):
        if 1:
            items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
            updated_inputs = []

            for user_input, recommended_item in zip(self.inputs[index], recommended_items):
                # Concatenate user_input with recommended item indices
                u_input = np.concatenate((user_input, [recommended_item]))
                updated_inputs.append(u_input)  # Append u_input to the list

                # Convert the list of u_input arrays to a NumPy array
            updated_inputs = np.array(updated_inputs)
            for u_input in updated_inputs:
                n_node.append(len(np.unique(u_input)))
            max_n_node = np.max(n_node)
            # print('maximum number of nodes')
            # print(max_n_node)
            if dataset == 'sample':
                max_n_items = 32
            elif dataset == 'diginetica':
                max_n_items = 82
            else:
                max_n_items = 146
            max_length = 0
            for u_input in updated_inputs:
                length = len(u_input)
                if length > max_length:
                    max_length = length

            for u_input in updated_inputs:
                node = np.unique(u_input)
                alias_inputs.append(
                    np.array([np.where(node == i)[0][0] for i in u_input]+(max_n_items-max_length) * [0]))

                items.append(node.tolist()+(max_n_items-len(node)) * [0])
                u_A = np.zeros((max_n_items, max_n_items))
                for i in np.arange(max_n_node):
                    if u_input[i+1] == 0:
                        break
                    u = np.where(node == u_input[i])[0][0]
                    v = np.where(node == u_input[i+1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)

                A_in.append(u_A_in)
                A_out.append(u_A_out)
            mask = self.mask[index]
            padded_mask = np.pad(mask, ((0, 0), (0, max_n_items-self.mask.shape[1])), mode='constant',
                                 constant_values=0)
            # print(f'self.targets[index]', self.targets[index])
            return A_in, A_out, alias_inputs, items, padded_mask, self.targets[index], max_n_node
