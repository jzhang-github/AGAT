#############
GatLib
#############


.. Note:: The AGAT APIs are here.

.. py:class:: EarlyStopping

   Stop training when model performance stop improving for some steps.

   .. py:method:: __init__(self, model, graph, logger, patience=10, folder='files')
   
      :param model model: The agat model object.
      :param DGL.graph graph: crystal graph.
      :param _io.TextIOWrapper logger: log file.
      :param int patience: stop patience. Stop training when model performance stop improving for some steps.
      :param str folder: directory for storing well-trained model


   .. py:property:: model
   
      AGAT model.
   
   
   .. py:property:: patience
   
      Patience steps.
   
   
   .. py:property:: counter
   
      Number of steps since last improvement of model performance.
   
   
   .. py:property:: best_score
   
      Best model performance.
   
   
   
   .. py:property:: update 
   
      Update state.
   
   
   .. py:property:: early_stop
   
      Stop training if this variable is true.
   
   
   .. py:property:: folder
   
      Directory for storing well-trained model.
   
   
   .. py:property:: _graph
   
      A crystal graph.
   
   
   .. py:method:: step(self, score, model)
   
      :param float score: metrics of model performance
      :param agat model: AGAT model object.
      
   
   .. py:method:: save_checkpoint(self, model)
   
      Save well-trained model (save checkpoint file) under :py:attr:`folder`.
      
      :param agat model: AGAT model object.
      
      
.. py:function:: forward(model, graph)

   Forward calculation by feeding one graph.
   
   .. Note:: Discarded


.. py:function:: forward_serial(model, graph, energy_list, forces_list, calculator)

   Forward calculation by feeding a graph.
   
   .. Important:: clone graphs before calling this function
   
   .. Note:: Discarded

.. py:function:: forward_parallel(model, graph_list, calculator)

   .. Note:: Discarded


.. py:function:: load_gat_weights(model, graph, ckpt_path, logger, device)

   Load trainable weights.
   
   :param agat model: a fresh GAT model without trainable `variables`.
   :param DGL.graph graph: a DGL graph
   :param str ckpt_path: Path to the saved checkpoint files
   :param _io.TextIOWrapper logger: log file.
   :param str device: calculation on CPU or GPU device.
   
   
      .. Hint:: - ``"/cpu:0"``: on CPU cores; 
         - ``"/gpu:0"``: On GPU card.

.. py:function:: accuracy(metrics, y_true, y_pred)

   .. Note:: Not used.
   

.. py:function:: PearsonR(y_pred, y_true)

   Calculate Pearson coefficient.

   :param tf.Tensor y_pred: predicted labels.
   :param tf.Tensor y_pred: true labels.
   :Returns: Pearson coefficient


.. py:function:: get_src_dst_data(graph)

   Extract node attribute: `h`
   
   .. Note:: Not used yet in the AGAT model.

   :param DGL.graph graph: crystal graph.
   :Returns:
      - src_data: data on source nodes.
      - dst_data: data on destination nodes.
      
      

