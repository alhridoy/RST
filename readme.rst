@util._wraps(np.arange, lax_description= """
.. note::

   Using ``arange`` with the ``step`` argument can lead to precision errors, 
   especially with lower-precision data types like ``fp8`` and ``bf16``. 
   For more details, see the docstring of :func:`numpy.arange`.
   To avoid precision errors, consider using an expression like 
   ``(jnp.arange(-600, 600) * .01).astype(jnp.bfloat16)`` to generate a sequence in a higher precision 
   and then convert it to the desired lower precision.
""")
