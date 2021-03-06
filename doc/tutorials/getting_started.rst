Overview
========

As of now, these tutorials assume a bleeding-edge version of AML. 

Download
--------

* `Development version <https://github.com/anlsys/aml>`_:

.. code-block:: console
  
  $ git clone --recurse-submodules https://github.com/anlsys/aml.git
  $ git checkout staging

Requirements:
-------------

* autoconf
* automake
* libtool
* libnuma

Installation
------------

.. code-block:: console
 
  $ sh autogen.sh
  $ ./configure
  $ make -j install

Compiling the Examples
----------------------

Each subdirectory in *doc/tutorials/* should contain a Makefile that can be
used to compile the examples and solutions to exercises.

