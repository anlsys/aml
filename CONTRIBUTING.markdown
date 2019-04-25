AML Contribution Guide {#contributing_page}
============

Welcome to AML contribution guide.
This page walk through the main library expectations.
When contributing to AML, contributors should write
features aligned with AML goal and AML spirit.
Additionnally, several good practices are requested, some beeing inforced
through the continuous integration (CI) pipeline.
For readability and maintainability, coding conventions need to be respected,
code needs to be tested, documented and integrated with the main AML branch
and its build tool-chain.

## General Guidelines

At its core, AML is able to provide access to established, well known
abstractions inside a common toolbox, with the more flexibility possible.
Abstractions are made available through __building blocks__, each representing
a single abstraction. Each building block is separated into a __generic__ API
for operations that the abstraction provides, and __implementations__ of those
operations for a specific implementation of the abstraction.
__generic__ API, i.e the first layer exposed to users
through a generic interface declares a table of operations common to the
abstraction, the main abstraction structure containing a such a table
and an opaque handle to store implementation specific data.

```
struct aml_building_block_ops{
       aml_building_block_op0();
       ...
};

struct aml_building_block_data;

struct aml_building_block{
       struct aml_building_block_ops *ops;
       struct aml_building_block_data *data;
};
```

A set of generic functions taking the abstraction as first argument
is supposed to be exposed while it will de-reference underlying
implementation function table to implement expected behavior.
Such generic functions are supposed to return an AML error code as defined
in [error header](@ref error.h).

```
int aml_building_block_feature(struct aml_building_block *, ...);
```

Building blocks should interact with each other through the generic API, but
specific implementations are allowed to break this rule for
performance-oriented reasons. A generic version should still be available.
It is expected that these will be transparent and composable,
because low-level optimizations require control.
Therefore, most implementation details (structures, functions) will be exposed
to the user in a dedicated header, and use of opaque structures, static functions
and variables there should be avoided.
Moreover, customization of a building block should happen at creation time: the goal
is to make it easy for users to evaluate different configurations options with
minimal impact on the inner code of the application.
Therefore, building blocks specialized implementation, must declare a dynamic allocator, free,
initialization and finish functions, and stack declaration.

```
int  aml_building_block_create (struct aml_building_block **, ...);
void aml_building_block_destroy(struct aml_building_block *);
int  aml_building_block_init   (struct aml_building_block *, ...);
void aml_building_block_fini   (struct aml_building_block *);

#define AML_BUILDING_BLOCK_DECL(name) \
	struct aml_building_block_data __ ##name## _inner_data; \
	struct aml_building_block name = { \
		&aml_building_block_ops, \
		(struct aml_building_block_data *)&__ ## name ## _inner_data, \
	}
```

Usually a building blocks implementation would declare an exportable (`extern`)
`struct aml_building_block_ops`, and `struct aml_building_block`, to provide
users with a default static implementation of the block.

Finally, building blocks headers, sources and tests go to a dedicated folder
located respectively in `include`, `src` and `tests`.

## Coding Convention

AML code respects several coding conventions. Some of them are enforced through
the CI pipeline.
Prior to start coding AML, you might want to set your editor to minimize
the process of code formatting.
Most of the formatting can also be scripted with `indent` program.
Code formatting compliance with AML requirements can be checked by running
[linux kernel linter](https://github.com/torvalds/linux/blob/master/scripts/checkpatch.pl)
at the root of AML project.

The list of coding convention is the following:
* use tabulation of 8 characters,
* remove trailing white spaces,
* lines may not exceed 80 characters,
* pointers star have to stick to the variable name: `void *var;`,
* function return type is on the same line as function name,
* curly braces are on the same line as the statement or function,
* A new line must appear after curly braces.
* macros are UPPERCASE_WITH_UNDERSCORES, function lowercase_with_underscores
* All functions/macros should start with aml/AML followed by the name of the building block involved.

## Versioning and Submission Workflow

AML code and CI runners is hosted on Argonne National Laboratory
[gitlab](https://xgitlab.cels.anl.gov/argo/aml) infrastructure under git version control.
New user may create new branches upon request to host and test their code.
New branched must be created from the master branch and merging is done
via git or gitlab pull request interface.

We basically follow the [Gitlab Flow](https://docs.gitlab.com/ee/workflow/gitlab_flow.html)
model:
 - `master` is the primary branch for up-to-date development
 - any new feature/development should go into a `feature-branch` based on
   `master` and merge into `master` after.
 - we follow a `semi-linear history with merge commits` strategy: each merge
   request should include clean commits, but only the HEAD and merge commit
   need to pass CI.
 - if the feature branch is not too complex, a single commit, even from a
   squash is okay. If the feature branch includes too many changes, each major
   change should appear in its own commit.
 - to help with in-development work, CI will not activate on branches with
   names started with wip/

### Commit Messages Styleguide

- use present tense, imperative mood
- reference issues and merge requests
- you can use [Gitlab Flavored Markdown](https://docs.gitlab.com/ee/user/markdown.html)

If you want some help, here is a commit template that you can add to your git
configuration. Save it to a `my-commit-template.txt` file and use `git config
commit.template my-config-template.txt` to use it automatically.

```
# [type] If applied, this commit will...    ---->|


# Why this change

# Links or keys to tickets and others
# --- COMMIT END ---
# Type can be 
#    feature
#    fix (bug fix)
#    doc (changes to documentation)
#    style (formatting, missing semi colons, etc; no code change)
#    refactor (refactoring production code)
#    test (adding missing tests, refactoring tests; no production code change)
# --------------------
# Remember to
#    Separate subject from body with a blank line
#    Limit the subject line to 50 characters
#    Capitalize the subject line
#    Do not end the subject line with a period
#    Use the imperative mood in the subject line
#    Wrap the body at 72 characters
#    Use the body to explain what and why vs. how
#    Can use multiple lines with "-" for bullet points in body
# --------------------
```

### Signoff on Contributions:

The project uses the [Developer Certificate of
Origin](https://developercertificate.org/) for copyright and license
management. We ask that you sign-off all of your commits as certification that
you have the rights to submit this work under the license of the project (in
the `LICENSE` file) and that you agree to the DCO.

To signoff commits, use: `git commit --signoff`.
To signoff a branch after the fact: `git rebase --signoff`

## Testing

AML project includes a set of unit and integration tests.
Each AML building block has its own set of tests in a separate directory.
Tests are launched together with automake test suite with the `make check` command.
Building blocks should be tested as thoroughly as possible and whenever
relevant, integration tests with other components should be available.
Tests compilation and launch are implemented together in `tests/Makefile.am`.
Whenever a branch is pushed to AML repository, it automatically triggers
the CI pipeline, unless branch name starts with `wip/`.
The CI pipeline must be successful for a merge request to be accepted.

## Documentation

For obvious reasons, the code and building block must be documented.
The whole documentatation is based doxygen tool.
All the public code (data structures, functions, macro, variables) must be
documented as precisely as possible. Additionnaly, new buiding blocks implementation
must create a group (with `\@defgroup`) and provide a detailed description of
the features it provides.  Syntax for code documentation can be found
[here](http://www.doxygen.nl/manual/docblocks.html),
and most useful elements can be found in the example below.

```
/**
 * \@defgroup aml_buidling_block_implementation "AML buidling block implemation"
 *
 * \@brief This is a short description of this group.
 *
 * The documentation encapsulated below
 * including this long description of the building_block
 * implementation will be create a section dedicated to this buidling block
 * implementation. It will be possible to link to it in the main header
 * documentation with "@ref" or "@see".
 * \@{
 **/

/**
 * This is a default operation table implementing
 * table defined in main header.
 **/
extern struct aml_building_block_ops aml_building_block_ops;

/**
 * This is the bb inner data declaration, to
 * stuff with needed attributes.
 **/
struct aml_building_block_data {};

/**
 * This is the bb declaration
 **/
struct aml_building_block{
       struct aml_building_block_ops *ops;
       struct aml_building_block_data *data;
};

/** This is the static buidling block declaration macro **/
#define AML_BUILDING_BLOCK_DECL(name) \
	struct aml_building_block_data __ ##name## _inner_data; \
	struct aml_building_block name = { \
		&aml_building_block_ops, \
		(struct aml_building_block_data *)&__ ## name ## _inner_data, \
	}

/** Static declaration of the building block size **/
#define AML_BUILDING_BLOCK_ALLOCSIZE (sizeof(struct aml_building_block_data) + \
				 sizeof(struct aml_building_block))


/**
 * Allocates and initializes a new building_block.
 *
 * \@param bb[out]: A pointer to store the new building block.
 * \@param other[in]: Other parameters used as input. 
 * \@return AML_SUCCESS if successful; an error code otherwise.
 **/
int aml_building_block_create(struct aml_building_block **bb, ...);

...

/**
 * This is the end of the inclusion into building_block documentation group.
 * \@}
 **/
```

If the contribution is an implementation of a building block and header has
been placed in `include/aml/building_block` then its documentation will automatically added to
the project documentation. However, it is necessary to point to this particular group
with `@ref` or `@see` command in the [main aml header](\ref aml.h).
If the contribution creates new include directories, the latter have to be included
in `doc/aml.doxy` after the field `INPUT`.

