AML: Tutorials
==============

This section contains step by step tutorials from each of the building blocks
of AML. They are intended to be followed in order, and include both
explanations of each building block abstraction as well as directed exercises
to better understand each abstraction.

Overview
--------

As of now, these tutorials assume a bleeding-edge version of AML. 

Download
~~~~~~~~

* `Development version <https://xgitlab.cels.anl.gov/argo/aml>`_:

.. code-block:: console
  
  $ git clone git@xgitlab.cels.anl.gov:argo/aml.git
  $ git checkout staging

Requirements:
~~~~~~~~~~~~~

* autoconf
* automake
* libtool
* libnuma

Installation
~~~~~~~~~~~~

.. code-block:: console
 
  $ sh autogen.sh
  $ ./configure
  $ make -j install

