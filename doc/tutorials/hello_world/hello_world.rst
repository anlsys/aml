Hello World: Init and Version Check
===================================

A first and easy test that AML is setup properly and can be linked with a user
program is to try to check that headers and library version are matching.  All
AML programs must also start by initializing the library, and must end with a
call to the cleanup function.

APIs
-------

Setup/Teardown API
~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: aml_init

.. doxygenfunction:: aml_finalize

Version API
~~~~~~~~~~~

.. doxygengroup:: aml_version

Usage
-----

Both setup and version APIs is available directly from the main AML header.

.. code-block:: c
  
  #include <aml.h>

Initialization is done by passing pointers to the command line arguments of
the program to the library.


.. code-block:: c
  
  int main(int argc, char **argv){
  ...
  if(aml_init(&argc, &argv) != 0){
    fprintf(stderr, "AML library init failure!\n");
    return 1;
  }

Checking the version is as easy as comparing the header version
*AML_VERSION_MAJOR* and the library-embedded version *aml_version_major*.

.. code-block:: c
  
  if(aml_version_major != AML_VERSION_MAJOR){
      fprintf(stderr, "AML ABI mismatch!\n");
      return 1;
  }

Exercise
--------

Write a "hello world" code that checks both major and minor versions of the
code.

You can find the solution in *doc/tutorials/hello_world*.

.. literalinclude:: 0_hello.c
   :language: c
