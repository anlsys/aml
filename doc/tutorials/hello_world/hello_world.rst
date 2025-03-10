Hello World: Init and Version Check
===================================

A first and easy test that AML is set up properly and can be linked with a user
program is to try to check that the headers and the library version match.  All
AML programs must also start by initializing the library and must end with a
call to the cleanup function.

APIs
-------

Setup/Teardown API
~~~~~~~~~~~~~~~~~~

.. doxygengroup:: aml
   :content-only:


Version API
~~~~~~~~~~~

AML defines its version using semantic versioning, with a macro and variable
for each subcomponent of the version number:

- **Version major**: denotes ABI changes which prevent compatibility with previous
  major version ABI.
- **Version minor**: denotes new features or improvement without breaking the old
  ABI.
- **Patch version**: patch and fix releases only.

The full version string is also available directly.

.. code-block:: c

  /* see aml/utils/version.h for details */
  #define AML_VERSION_MAJOR  ...
  #define AML_VERSION_MINOR  ...
  #define AML_VERSION_PATCH  ...
  #define AML_VERSION_STRING ...

  extern const int aml_version_major    = ...;
  extern const int aml_version_minor    = ...;
  extern const int aml_version_patch    = ...;
  extern const char *aml_version_string = ...;


Usage
-----

Both the setup and the version APIs are available directly from the main AML
header.

.. code-block:: c
  
  #include <aml.h>

Initialization is done by passing pointers to the command-line arguments of
the program to the library.

.. code-block:: c
  
  int main(int argc, char **argv){
  ...
  if(aml_init(&argc, &argv) != 0){
    fprintf(stderr, "AML library init failure!\n");
    return 1;
  }

Checking that the program is linked against the same version as the one used
during compilation is as easy as comparing the header version
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

.. container:: toggle

   .. container:: header

      **Click Here to Show/Hide Code**

   .. literalinclude:: 0_hello.c
      :language: c

You can find this solution in *doc/tutorials/hello_world/*.
