#############################
PymatgenStructureAnalyzer
#############################

.. Note:: Adopted from pymatgen. get_connections() are replaced by get_connections_new(). The new one is faster and more suitable for constructing a DGL graph.


Here is the code of ``get_connections_new(self)``.

.. code-block:: python

   def get_connections_new(self):     # modified by ZHANG Jun on 20210421, to build dgl graph
    """
    Returns a list of site pairs that are Voronoi Neighbors, along
    with their real-space distances.
    After test, this function returns the correct connections for a
    supercell big enough.
    """
    sender, receiver, dist = [], [], []
    maxconn = self.max_connectivity
    for ii in range(0, maxconn.shape[0]):
        for jj in range(ii + 1, maxconn.shape[1]):
            if maxconn[ii][jj] != 0:
                dist.append(self.s.get_distance(ii, jj))
                sender.append(ii)
                receiver.append(jj)
    # print(np.shape(sender), np.shape(receiver), np.shape(dist))
    bsender   = sender   + receiver      # bidirectional
    breceiver = receiver + sender        # bidirectional
    bsender  += [x for x in range(maxconn.shape[0])] # add self_loop
    breceiver+= [x for x in range(maxconn.shape[0])] # add self_loop
    dist     *= 2                        # bidirectional
    dist     += [0.0] * maxconn.shape[0] # add self loop for `dist`
    # print(np.shape(bsender), np.shape(breceiver), np.shape(dist))
    dist      = [[x] for x in dist]
    
    return bsender, breceiver, dist


.. function:: get_connections_new()

   Returns a list of site pairs that are Voronoi Neighbors, along with their real-space distances.
   
   :Returns: - bsender: index of senders.
      - breceiver: index of receivers.
      - dist: distance between senders and receivers.