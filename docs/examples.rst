.. _examples:

#############
Usage example
#############

Here's a quick example of how to pass 2D pupil detection results to pye3d (requires
standalone `2D pupil detector <https://github.com/pupil-labs/pupil-detectors/>`_
installation). Alternatively, you can install the dependencies together with ``pye3d``

.. code-block:: console

   pip install pye3d[examples]

.. literalinclude:: ../examples/process_eye_video.py
   :language: python
   :linenos:
   :emphasize-lines: 11,13,14,26,27,29
