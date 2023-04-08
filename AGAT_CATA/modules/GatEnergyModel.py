
import tensorflow as tf
from tensorflow.keras import Model as tf_model
from modules.SingleGatLayer import GATLayer
from dgl.ops import edge_softmax

class GAT(tf_model):
    """
    Description:
    ----------
        A GAT model with multiple gat layers.
    Parameters
    ----------
    num_gat_out_list: list
        A list of numbers that contains the representation dimension of each GAT layer.
    num_readout_out_list: list
        A list of numbers that contains the representation dimension of each readout layer.
    head_list_en: list
        A list contains the attention mechanisms of each head for the energy prediction.
    embed_activation: str
        TensorFlow activation function for embedding the inputs.
    readout_activation: str
        TensorFlow activation function for the readout layers.
    bias: boolean
        bias term of dense layers.
    negative_slope: float
        Negative slope of LeakyReLu function.
    Return of `call` method
    -----------------------
    en_h : tf Tensor
        Raw energy predictions of each atom (node).
    Important note
    -----------------------
    The last readout list must be one. Because the node energy or node force should have one value.
    """

    def __init__(self,
                 num_gat_out_list,
                 num_readout_out_list=[1],
                 head_list_en=['div'],
                 embed_activation='LeakyReLU',
                 readout_activation='LeakyReLU',
                 bias=True,
                 negative_slope=0.2):
        super(GAT, self).__init__()

        # the dimension of input features will be determined automatically.
        self.num_gat_out_list     = num_gat_out_list
        self.num_gat_layer        = len(num_gat_out_list)
        self.num_readout_out_list = num_readout_out_list
        self.num_readout_layer    = len(num_readout_out_list)

        # embedding inputs in every gat layer, which is more flexible than embeding inputs here.

        self.head_list_en    = head_list_en
        self.num_heads_en    = len(head_list_en)
        self.negative_slope  = negative_slope
        self._activation_f   = {'LeakyReLU':    tf.keras.layers.LeakyReLU(alpha=self.negative_slope),
                                'relu':         tf.keras.activations.relu,
                                'tanh':         tf.keras.activations.tanh,
                                'none':         None,
                                'softmax':      tf.keras.activations.softmax,
                                'edge_softmax': edge_softmax}

        self.embed_act_str, self.readout_act_str = [embed_activation, readout_activation]
        self.embed_act       = self._activation_f[embed_activation]

        self.readout_act     = self._activation_f[readout_activation]
        self.bias            = bias

        self.head_fn         = {'mul' : self.mul,
                                'div' : self.div,
                                'free': self.free}

        self.gat_layers         = []
        self.en_read_out_layers = []
        self.batch_normalization = False

        # GAT layer
        for l in range(self.num_gat_layer):
            self.gat_layers.append(GATLayer(self.num_gat_out_list[l],
                                            self.num_heads_en,
                                            bias=self.bias,
                                            negative_slope=self.negative_slope,
                                            activation=self.embed_act))

        # energy readout layer
        for l in range(self.num_readout_layer-1):
            self.en_read_out_layers.append(tf.keras.layers.Dense(self.num_readout_out_list[l],
                                                                    self.readout_act,
                                                                    self.bias))
        self.en_read_out_layers.append(tf.keras.layers.Dense(self.num_readout_out_list[-1],
                                                                self._activation_f['none'],
                                                                self.bias))

    def mul(self, TfTensor):
        return TfTensor

    def div(self, TfTensor):
        return 1/TfTensor

    def free(self, TfTensor):
        return tf.constant(1.0, shape=TfTensor.shape)

    def get_head_mechanism(self, fn_list, TfTensor):
        """
        :param fn_list: A list of head mechanisms. For example: ['mul', 'div', 'free']
        :type fn_list: list
        :param TfTensor: A tensorflow tensor
        :type TfTensor: tf.tensor
        :return: A new tensor after the transformation.
        :rtype: tensor

        """
        TfTensor_list = []
        for func in fn_list:
            TfTensor_list.append(self.head_fn[func](TfTensor))
        return tf.concat(TfTensor_list, 1)

    def call(self, graph):
        '''
        Description:
        ----------
            `call` function of GAT model.
        Parameters
        ----------
        graph: `DGL.Graph`
            A graph.
        Return
        -----------------------
        en_h : tf Tensor
            Raw energy predictions of each atom (node).
        '''
        with graph.local_scope():
            h    = graph.ndata['h']                                    # shape: (number of nodes, dimension of one-hot code representation)
            dist = tf.reshape(graph.edata['dist'], (-1, 1, 1))         # shape: (number of edges, 1, 1)
            dist = tf.where(dist < 0.5, 0.5, dist)                     # This will creat a new `dist` variable, insted of modifying the original memory.
            dist = self.get_head_mechanism(self.head_list_en, dist)    # shape of dist: (number of edges, number of heads, 1)

            for l in range(self.num_gat_layer):
                h = self.gat_layers[l](h, dist, graph)                 # shape of h: (number of nodes, number of heads * num_out)

            # read-out layers
            for l in range(self.num_readout_layer):
                h = self.en_read_out_layers[l](h)
            return h

# debug
if __name__ == '__main__':
    model = GAT([30,40,50],
                 num_readout_out_list=[1],
                 head_list_en=['div', 'mul'],
                 embed_activation='LeakyReLU',
                 readout_activation='LeakyReLU',
                 bias=True,
                 negative_slope=0.2)

    from modules.Crystal2Graph import CrystalGraph
    cg = CrystalGraph()

    bg = cg.get_graph('POSCAR.txt', super_cell=False)
    en = model(bg).numpy()


