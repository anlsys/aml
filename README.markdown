Argonne's Memory Library
========================

This is a prototype library to manage the hierarchical memory present in recent
and future HPC systems, including NVM and stacked memory.

This library defines "memory nodes" as hardware locations where data can
reside. Not all of those memory nodes can provide virtual memory access to the
data they contain, and as such, allocations are identified only by a UID.
Memory nodes are defined by a public data structure that also contains
information on the granularity of data accesses to a node.

The library then provides functions to "pull" data into a virtual memory
location. That memory location should be seen as temporary: the user should ask
the library to "push" out the data afterwards.
