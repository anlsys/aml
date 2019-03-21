Overall Design and Coding Guidelines
====================================

The AML codebase try to follow a small set of overarching principles in its
design, as well as some more specific code style constraints.

## Overal Design Principles

At its core, AML is able to provide access to established, well known
abstractions inside a common toolbox, with the more flexibility possible.
Abstractions are made available through __building blocks__, each representing
a single abstraction. Each building block is separated into a __generic__ API
for operations that the abstraction provides, and __implementations__ of those
operations for a specific implementation of the abstraction.

The following principles apply when designing abstractions and implementations:

1. Abstractions should be kept as minimal and as separated as possible: if two
different functionalities seem to involve different kinds of concepts that
could be dealt with or configured independently, they should be split into
different building block

2. Customization of a building block should happen at creation time: the goal
is to make it easy for users to evaluate different configurations options with
minimal impact on the inner code of the application

3. Creating a new implementation of a building block is preferred to adding
complexity to an existing implementation.

4. Building blocks should interact with each other through the generic API, but
specific implementations are allowed to break this rule for
performance-oriented reasons. A generic version should still be available.

5. Its okay for an implementation not to work unless the building block it
depends on are from specific implementations. If possible, checks and error
codes should be used to enforce correctness.

6. Neither the opaque structures used for the generic API nor the actual
structures used by implementations are private, as users should be able to
customize any of those on the fly. That goes also for functions.

## Coding Guidelines

1. no `static` functions inside implementations, users should be able to create
their own custom building block by combining existing code.

2. no `const` pointers to functions or other objects, for the same reasons.

3. no `void` return type, unless no implementation can ever fail.

4. macros are UPPERCASE_WITH_UNDERSCORES, function lowercase_with_underscores

5. All functions/macros should start with aml/AML followed by the name of the
building block involved.
