Installation
============

Install the custom forks
------------------------

Install the ``EdinhoAndra/bnlearn-cpp`` fork directly from GitHub:

.. code-block:: console

    pip install "bnlearn @ git+https://github.com/EdinhoAndra/bnlearn-cpp.git@master"

For CUDA 12, install its GPU extra:

.. code-block:: console

    pip install "bnlearn[gpu-cu12] @ git+https://github.com/EdinhoAndra/bnlearn-cpp.git@master"

The GPU extra includes CuPy and the CUDA 12 runtime components and headers
needed by the runtime-compiled CUDA C++ kernel. A compatible NVIDIA driver is
still required.

The package metadata installs ``pgmpy`` exclusively from
``https://github.com/EdinhoAndra/pgmpy-cpp.git``. Do not install the PyPI
versions of ``bnlearn`` or ``pgmpy`` into the same environment.

Create Environment
------------------

For better dependency management, it's recommended to install ``bnlearn`` in an isolated Python environment using conda:

.. code-block:: console

    conda create -n env_bnlearn python=3.13
    conda activate env_bnlearn

.. _installation step 1:

.. figure:: ../figs/01_installation.png

   Create a new conda environment.

After activation, your command prompt should show the environment name. For example:

.. code-block:: console

   (env_bnlearn) D:\>

Uninstall
---------

To remove the ``bnlearn`` installation and its environment:

.. code-block:: console

   # List all active environments
   conda env list

   # Remove the bnlearn environment
   conda env remove --name env_bnlearn

   # Verify removal by listing environments again
   conda env list

Validate Installation
---------------------

To verify your installation, start Python in your console:

.. code-block:: console

   python

Then run the following code, which should generate a figure:

.. code-block:: python

   import bnlearn as bn
   df = bn.import_example()
   model = bn.structure_learning.fit(df)
   G = bn.plot(model)

.. _installation step 4:

.. figure:: ../figs/04_installation.png

Troubleshooting Import Errors
-----------------------------

If you're using Jupyter Notebook or Google Colab, you might encounter a NumPy version compatibility error:

.. code-block:: python

    import bnlearn as bn
    # Error message:
    RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
    ImportError: numpy.core.multiarray failed to import

This error occurs because ``bnlearn`` requires NumPy version 1.24.1 or higher. To resolve this:

1. To fix this, you need an installation of *numpy version=>1.24.1* which is installed during the ``bnlearn`` installation.
   However, when you are using colab or a jupyter notebook, you need to reset your kernel first to let it work. 
2. If using Colab or Jupyter Notebook:
   - Go to the menu
   - Click **Runtime -> Restart runtime**
   - Re-import bnlearn


.. include:: add_bottom.add
