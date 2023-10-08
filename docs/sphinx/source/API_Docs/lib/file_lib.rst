file_lib
#########


 

.. function:: generate_file_name(fname)

   Generate a new file name according to the input.

   :param str fname: file name.
   :Returns: fname. A new file name by appending `_new`
   :rtype: str


.. function:: file_exit()

   Stop high-throughput prediction if ``StopPython`` is detected.
   
   :raises FileExit: Exit because `StopPython` file is found.
  
  
  
.. function:: modify_INCAR(key='NSW', value='300', s='')

   Modify the INCAR file.
   
   Example::
   
      from agat.lib import modify_INCAR
      modify_INCAR('NSW', '200')
      

   :param key: The INCAR tag, defaults to 'NSW'. The INCAR tag must be found in :py:data:`agat.lib.incar_tag.INCAR_TAG`.
   :type key: str, optional
   :param value: Value of the INCAR tag, defaults to '300'
   :type value: str, optional
   :param s: Comment string, defaults to ''
   :type s: str, optional
   :return: Modified INCAR file
