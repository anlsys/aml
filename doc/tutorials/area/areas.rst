Areas
=====

AML areas represent places where data can be stored. In shared memory systems,
locality is a major concern for performance. Being able to query memory from
specific places is of major interest to achieve this goal. AML areas provide
low-level mmap() / munmap() functions to query memory from specific places
materialized as areas. Available area implementations dictate the way such
places can be arranged and their properties. It is important to note that the
functions provided through the Area API are low-level and are not optimized for
performance as allocators are.

Default Areas
-------------

Allocating Memory From an Area
------------------------------

Exercise: allocating interleaved pages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA Area
---------

Exercise: CUDA allocation
~~~~~~~~~~~~~~~~~~~~~~~~~


Implementing a Custom Area
--------------------------

Exercise: interleaving in blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
