file_lib
#########


.. function:: file_exit()

   Stop high-throughput prediction if ``StopPython`` is detected.
   
   :raises FileExit: Exit because `StopPython` file is found.
   

.. function:: generate_file_name(fname)

   Generate a new file name according to the input.

   :param str fname: file name.
   :Returns: fname. A new file name by appending `_new`
   :rtype: str
