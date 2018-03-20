AML: Building Blocks for Memory Management
=========================================

AML is a library to manage byte-addressable memory devices. The library
is designed as a collection of building blocks, so that users can create custom
memory management policies for allocation and placement of data across devices.

This library is still in the prototyping phase. APIs might break often.

# General Architecture

The architecture of the library relies on two principles: type-specific
initialization functions, and generic interfaces to each building block.

The type-specific initialization functions include:
  - `_DECL` macros to declare a typed building block on the stack.
  - `_create_` functions to allocate a building block and return its pointer.
  - `_init_` functions to initialize stack-allocated building blocks.

Generic interfaces all take a pointer to a building block.

# Low-Level Building Blocks

Low-level building blocks provide the basic mechanisms required to implement
any high-level memory management policy. This include:
  - *arenas:* allocation policies inside a memory region
  - *areas:* actual memory reservation on devices
  - *bindings:* mapping pages/tiles of memory unto multiple devices
  - *tilings:* chunking data

# High-Level Building Blocks

High-level building blocks use the low-level ones to provide fancier memory
management facilities, including:
  - *dmas:* moving data across areas
  - *scratchpads:* using an area as an explicitly managed cache of another one
