.. _install_from_source:

#############################
Build and install from source
#############################

Follow these instructions if you want to install a modified version of the source code.
Otherwise, we recommend installing the pre-packaged binary wheels from PyPI:

.. code-block:: console

   pip install pye3d

To install from source, see the table of contents on the right.

Installing using Conda
#########################

`Conda <https://docs.conda.io/en/latest/index.html>`_ makes it easy to install
``pye3d``'s non-Python dependencies.

Creating a new Conda environment
--------------------------------

The ``conda-env.yml`` file defines all dependencies and will install ``pye3d`` as part
of the environment creation.

.. code-block:: console

   # Download pye3d code
   git clone https://github.com/pupil-labs/pye3d-detector.git
   cd pye3d-detector

   conda env create --quiet --name <new environment name> --file conda-env.yml
   conda activate <new environment name>

Using an existing Conda environment
-----------------------------------

If you want to reuse an existing environment, you need to install the dependencies
and ``pye3d`` manually using the commands below.

.. code-block:: console

   conda activate <existing environment>
   conda install -c conda-forge eigen

   git clone https://github.com/pupil-labs/pye3d-detector.git
   cd pye3d-detector

   pip install -e .

.. include:: known_build_issue.rst

Installing everything from scratch
##################################

.. seealso::

   The pye3d Github Actions `build pipeline`_ uses this methodology.

.. _build pipeline: https://github.com/pupil-labs/pye3d-detector/actions/workflows/build-pye3d.yml

Build dependencies
------------------

You can skip this step if you have OpenCV and Eigen3 compiled and installed on your computer.
Scroll to the bottom and continue with installing pye3d.

Windows
^^^^^^^

Building the dependencies on Windows requires running the commands in [PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/overview). Requires `git` and `cmake` to be in your system PATH. Please run all three install steps in the same shell or redefine `OpenCV_DIR` and `Eigen3_DIR` before running the last step (building and installing pye3d).

Build and install Eigen
"""""""""""""""""""""""

.. code-block:: powershell

   # Download Eigen
   Invoke-WebRequest "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip" -OutFile eigen.zip

   # Prepare build
   Expand-Archive eigen.zip
   mv eigen/eigen-3.*/* eigen/
   mkdir eigen/build

   # Enter build path
   cd eigen/build

   # Configure build
   cmake .. -A x64 -DCMAKE_INSTALL_PREFIX="."

   # Compile
   cmake --build . --target INSTALL --config Release --parallel

   # Define Eigen3 location for third step
   $Env:Eigen3_DIR = (pwd)

   # Exit build path
   cd ../..

Go to :ref:`pyed_source_build_install` for the last step.

Ubuntu
^^^^^^

Build and install Eigen
"""""""""""""""""""""""

.. code-block:: bash

   # Download Eigen
   wget -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

   # Prepare build
   unzip eigen.zip
   mv eigen-3.* eigen/
   mkdir eigen/build

   # Enter build path
   cd eigen/build

   # Configure build
   cmake .. \
      -DCMAKE_INSTALL_PREFIX="."

   # Compile
   make
   make install

   # Define Eigen3 location for third step
   Eigen3_DIR=${pwd}

   # Exit build path
   cd ../..

Go to :ref:`pyed_source_build_install` for the last step.

macOS
^^^^^

Downlaoding the dependencies requires ``wget``, which can be installed on macOS with
`Homebrew <https://brew.sh>`_:

.. code-block:: bash

   brew install wget

Build and install Eigen
"""""""""""""""""""""""

.. code-block:: bash

   # Download Eigen
   wget -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

   # Prepare build
   unzip eigen.zip
   mv eigen-3.* eigen/
   mkdir eigen/build

   # Enter build path
   cd eigen/build

   # Configure build
   cmake .. -DCMAKE_INSTALL_PREFIX="."

   # Compile
   make
   make install

   # Define Eigen3 location for third step
   Eigen3_DIR=${pwd}

   # Exit build path
   cd ../..

See below for the last step.

.. _pyed_source_build_install:

Build and install ``pye3d`` (all platforms)
-------------------------------------------

Requires ``Eigen3_DIR`` environmental variable pointing to the appropriate install
locations.

.. code-block:: bash

   # Download pye3d code
   git clone https://github.com/pupil-labs/pye3d-detector.git

   # Prepare build
   cd pye3d-detector
   # optional: activate Python virtual environment here

   # Make sure pip is up-to-date
   python -m pip install -U pip

   # Build and install pye3d
   python -m pip install .

.. include:: known_build_issue.rst
