AML Contribution Guide {#contributing_page}
============

Welcome to AML contribution guide.

### Table Of Contents

1. [General Guidelines](#guidelines)
2. [Coding Conventions](#conventions)
3. [Versioning](#versioning)
4. [Submission Workflow](#workflow)
5. [Testing](#testing)
6. [Documenting AML](#documentation)
7. [Check-List Before Submitting Code](#checklist)

This page walks through the library expectations. When contributing to AML, 
contributors should write features aligned with AML goal and AML spirit.
Additionally, several good practices are requested, some being enforced
through the continuous integration (CI) pipeline.  For readability and
maintainability, coding conventions need to be respected, code needs to be
tested, documented and integrated with the main AML branch and its build
tool-chain.

## General Guidelines <a name="guidelines"></a>

### Building Blocks of AML

At its core, AML is able to provide access to established, well known
abstractions inside a common toolbox, with as much flexibility as possible.
Abstractions are made available through __building blocks__, each representing
a single abstraction. Each building block is separated into a __generic__ API
for operations that the abstraction provides, and __implementations__ of those
operations for a specific implementation of the abstraction.
The __generic__ API, i.e the first layer exposed to users
through a generic interface, declares a table of operations common to the
abstraction, the main abstraction structure containing a such a table
and an opaque handle to store implementation specific data.

```
struct aml_building_block_data;

struct aml_building_block_ops{
       type (*aml_building_block_op0)(struct aml_building_block_data *, args);
       ...
};

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
performance-oriented reasons. A generic version should still be available.  It
is expected that these will be transparent and composable, because low-level
optimizations require control.  Therefore, most implementation details
(structures, functions) will be exposed to the user in a dedicated header, and
use of opaque structures, static functions and variables there should be
avoided.  Moreover, customization of a building block should happen at creation
time: the goal is to make it easy for users to evaluate different
configurations options with minimal impact on the inner code of the
application.  Therefore, building blocks specialized implementation, must
declare a dynamic allocator, and free functions.

```
int  aml_building_block_create (struct aml_building_block **, ...);
void aml_building_block_destroy(struct aml_building_block *);

```

Usually a building blocks implementation would declare an exportable (`extern`)
`struct aml_building_block_ops`, and `struct aml_building_block`, to provide
users with a default static implementation of the block.

### AML Directory Organization

+ [include](./include):   
  The headers directory. All library sources headers and installed headers 
  go here.
  * [include/aml.h](./include/aml.h):   
  The main library header.
  This header must include all building block definition and includes all 
  `utils` headers.
  * [include/aml/](./include/aml/):   
  This directory includes abstraction implementation specific headers,
  higher level abstraction headers and library utils headers.
  * `include/aml/<building-block>/<header>.h`:   
  For each building blocks and there implementation, a header file is defined.
  * [include/aml/utils](./include/aml/utils):   
  This directory contains library and user utility headers.
  These headers must all be included in [include/aml.h](./include/aml.h).
	  - `include/aml/utils/<header>.h`:   
	  Library util headers (non backend dependent).
	  - `include/aml/utils/backend/<header>.h`:   
	  Library backend dependent headers. These header should be included with
	  appropriate macro guard as found in `include/aml/utils/features.h`.
  * [include/aml/higher/](./include/aml/higher/):   
	Directory for aml high level features that are not building blocks.
	  - `include/aml/higher/<abstraction-header>.h`:   
	  Higher level abstraction declares a generic high level interface.
	  - `include/aml/higher/<abstraction>/<header>.h`:   
	  Higher level abstraction declares an implementation of their high level 
	  interface.
+ [src](./src):   
  The library sources directory.
  * [src/aml.c](./src/aml.c):   
  Library initialization/cleanup source.
  * `src/<building-block>/<source>.c`:   
  Building blocks implementation.
  * `src/backend/<source>.c`:   
  Backends initialization/cleanup and related utility functions.
  * `src/utils/<source>.c`:   
  Implementation of library boilerplate code.
  * `src/<higher>/<source>.c`:   
  Implementations of non building block abstractions.
+ [tests](./tests):   
  This directory contains the library unit tests triggered in the CI by 
  `make check`.
  * `tests/<building-block>/test_<implementation>.c`:   
  Building blocks implementation specific tests.
  * `tests/<building-block>/test_<building-block>.c`:   
  Generic building block tests usable by all/most implementation tests.
  * `tests/<higher>/<abstraction>/test_<name>.c`:   
  High level (non building blocks) implementation specific and generic tests.
  * `tests/utils/test_<utils>.c`:   
  Utils tests.
+ [doc](./doc):   
  AML Documentation directory.
  See ["Documenting AML"](#documentation) section for more information on how
  to document AML properly.
  * [doc/pages](./doc/pages):   
	Textual pages of aml main header [include/aml.h](./include/aml.h).
	- `doc/pages/<basic-block>_<implementation>_api.rst`   
	Textual pages of each aml building block.
  * [doc/tutorials](./doc/tutorials):   
	Code tutorials directory.	
	- `doc/tutorials/<basic-block>/<num>_<name>.c`:   
	Solution source code of tutorials on how to use basic blocks.
	- `doc/tutorials/<basic-block>/<num>_<name>.trs`:   
	Matching textual exercises on how to use basic blocks.
	- `doc/tutorials/<higher>/<num>_<name>.c`:   
	Solution source code of tutorials on how to use higher level abstractions.
	- `doc/tutorials/<higher>/<num>_<name>.trs`:   
	Matching textual exercises on how to use higher level abstractions.

## Coding Conventions <a name="conventions"></a>

AML code respects several coding conventions, most of them enforced in the CI
pipeline using clang-format.

The list of coding convention is the following:
* use tabulation of 8 characters,
* remove trailing white spaces,
* lines may not exceed 80 characters,
* pointers star have to stick to the variable name: `void *var;`,
* function return type is on the same line as function name,
* curly braces are on the same line as the statement or function,
* A new line must appear after curly braces.
* macros are UPPERCASE_WITH_UNDERSCORES, function lowercase_with_underscores
* All functions/macros should start with aml/AML followed by the name of the 
building block involved.

### Editors Configuration

#### Emacs

Emacs can be configured to meet most formatting specified 
in [`.clang-format`](./.clang-format) file by adding the following 
configuration to  your `.emacs`.

```
;; Definition of aml C style.
(c-add-style "aml"
	     '((c-basic-offset . 8)
               (c-indent-tabs-mode . t)
	       (c-hanging-semi&comma-criteria (c-semi&comma-inside-parenlist))
	       (c-hanging-colon-alist .
				      ((case-label . (after))
				       (label . (after))
				       (statement-cont)
				       (statement)))
	       (c-hanging-braces-alist .
				       ((defun-open . (before))
					(defun-open . (after))
					(defun-block-intro . (after))
					(statement-block-intro . (after))
					(arglist-cont-nonempty . (after))
					(arglist-intro . (after))
					(topmost-intro . (after))
					(statement . (after))
					(substatement . (after))
					(else-clause . (after))
					(brace-list-open . (after))
					(brace-list-close . (after))
					(block-open . (after))
					(block-close . (after))))
	       (add-to-list 'c-cleanup-list 'brace-else-brace
			    'brace-elseif-brace 'empty-defun-braces
			    'defun-close-semi)
	       (c-offsets-alist . ((topmost-intro . 0)
				   (comment-intro . 0)
				   (inclass . +)
				   (case-label . 0)
				   (else-clause . 0)
				   (inextern-lang . 0)
				   (label . [0])
				   (arglist-cont-nonempty . c-lineup-arglist)
				   (class-close . c-lineup-close-paren)
				   (defun-close . c-lineup-close-paren)
				   (brace-list-close . c-lineup-close-paren)
				   (block-open . 0)
				   (block-close . c-lineup-close-paren)
				   (defun-block-intro . +)
				   (defun-open . 0)
				   (statement-case-intro . +)
				   (statement-block-intro . +)
				   (statement . 0)
				   (brace-list-intro . +)
				   (brace-list-entry . 0)
				   (substatement-open . 0)
				   (substatement . +)))))

;; Automatically set aml style when working with aml C files.
(defun aml-style ()
  (require 'cc-mode)
  (when
      (and buffer-file-name
	   (string-match "aml/" buffer-file-name))
    (c-set-style "aml")))
(add-hook 'c-mode-hook 'aml-style)
```

However the best method to have CI compliant formatting is to install 
the `clang-format` package and to use it to format your buffer. 
Below sample code uses `use-package` package to load `clang-format` and 
bind `C-x <tab>` to format selected region.

```
(use-package clang-format
  :commands
  (clang-format-region clang-format-buffer)
  :bind (("C-x <tab>" . 'clang-format-region)))
```

## Versioning <a name="versioning"></a>

AML uses the [semver](https://semver.org/) versionning standard to version the 
library. In a nutshell, AML versionning is composed of three non negative 
itegers `<x>.<y>.<z>` increasing as more code gets merged into the master 
branch. New features, bug fixes, performance and improvements shall raise 
the `<z>` number. Additions to the API which do not break the existing API
shall raise the `<y>` number. Whereas modifications of the library that alter 
existing API shall raise the `<x>` number. AML do not define yet an ABI.
The version of AML can be modified in [configure.ac](./configure.ac).

## Submission Workflow <a name="workflow"></a>

AML code is hosted on [github](https://github.com/anlsys/aml) and uses github 
actions as a CI. External collaborators (i.e. not members of the group at
Argonne National Laboratory) should rely on forking and pull requests from
outside the main repository. Internal members should push to new branches from
master and use pull requests to merge their work in it.

We previously followed the 
[Gitlab Flow](https://docs.gitlab.com/ee/workflow/gitlab_flow.html)
models, so the git history might still look like it if you look at previous
versions. We now follow a simplified process, but not the popular git-flow.
Basically:
 - `master` is the primary branch for up-to-date development
 - any new feature/development should go into a `feature-branch` based on
   `master` and merge into `master` after.
 - we follow a `semi-linear history with merge commits` strategy: each merge
   request should include clean commits, but only the HEAD and merge commit
   need to pass CI.
 - if the feature branch is not too complex, a single commit, even from a
   squash is okay. If the feature branch includes too many changes, each major
   change should appear in its own commit.

### Commit Messages Styleguide

- use present tense, imperative mood
- reference issues and merge requests

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

## Testing <a name="testing"></a>

The AML project includes a set of unit and integration tests.

### Unit tests:

AML unit tests are located in the [tests](./tests) directory.
Each AML building block has its own set of tests in a separate directory 
(`tests/<building-block>/test_<impl>.c`). The same apply to utils 
(`tests/<utils>/test_<util>.c`) and high level abstractions 
(`tests/<higher>/test_<impl>.c`). Any building block implementation,
high level abstraction implementation, utils, etc. should be tested with
a unit test. Tests are launched together with automake test suite with 
the `make check` command. Building blocks should be tested as thoroughly as 
possible and whenever relevant, integration tests with other components should 
be available. Tests compilation and launch are implemented together 
in `tests/Makefile.am`. 

### Integration Tests

* Tutorials:   
AML Tutorials are linked with the compiled library and run basic library uses.
Tutorial should be implemented at least for building blocks and high level 
abstractions. These tests also run with the `make check` tests and the CI.

* XSBench:   
Some AML features have been integrated in the 
[XSBench](https://github.com/ANL-CESAR/XSBench) proxy application and the 
application itself is built and ran in a Continuous Integration check.
   
   
**The CI pipeline must be successful for a merge request to be accepted.**

## Documenting AML <a name="documentation"></a>

For obvious reasons, the code and building block must be documented.
The whole documentation is based on doxygen tool.
All the public code (data structures, functions, macro, variables) must be
documented as precisely as possible. Additionally, new building blocks 
implementation must create a group (with `\@defgroup`) and provide a detailed 
description of the features it provides.  Syntax for code documentation can be 
found [here](http://www.doxygen.nl/manual/docblocks.html),
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
been placed in `include/aml/<building_block>/` then its documentation will 
automatically added to the project documentation. However, it is necessary to 
point to this particular group with `@ref` or `@see` command in 
the [main aml header](\ref aml.h). If the contribution creates new include 
directories, the latter have to be included in `doc/aml.doxy` after the field 
`INPUT`.

## Before Submitting Code <a name="checklist"></a>

The checklist below should solve most of the submission issues related to
automatic testing and enforcement of contribution guidelines.

* Rebase your work on master branch,
* Squash your branch and split it in one or several commits representing 
logical units of work. Do not include formatting of code you did not intend to 
modify in your pull-request because it will clutter the diff and make the review
cumbersome.
* Make sure the license is included in all the new source files.   
This can be automated as follow:
```
FILES=$(git ls-files | grep -E ".*\.c$|.*\.h$")
LICENSE_FILE=SPDX.txt
cat > $LICENSE_FILE <<EOF
 * Copyright 2019 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the AML project.
 * For more info, see https://github.com/anlsys/aml
 *
 * SPDX-License-Identifier: BSD-3-Clause
EOF
license_num_lines=$(awk 'END{print NR}' $LICENSE_FILE)

for f in ${FILES[@]}; do
	match=$(awk -v max="$license_num_lines" 'BEGIN{c=0}NR==FNR{a[$1];next}$1 in a{c++}c>=max{exit}END{print c}' $LICENSE_FILE $f)
	if [ "$match" != "$license_num_lines" ]; then
		head -n ${license_num_lines} $f
		printf "\tFix %s ? (y/n): " $f
		read ans
		if [ "$ans" == "y" ]; then
			cp $f $f.tmp
			echo "/*******************************************************************************" > $f
			cat $LICENSE_FILE >> $f
			echo " ******************************************************************************/" >> $f
			echo "" >> $f
			cat $f.tmp >> $f
			rm $f.tmp
		fi
	fi
done
rm $LICENSE_FILE
```

* Make sure all the new code (only) is properly formatted.   
This can be automated as follow:
```
FILES=$(git ls-files | grep -E ".*\.c$|.*\.h$")
diff_file=clang-format-diff
target=$(git rev-parse master)

for f in ${FILES[@]}; do
	rm -f $diff_file
	git-clang-format --quiet --diff $target -- $f > $diff_file
	lint=$(grep -v --color=never "no modified files to format" $diff_file || true)
	rm -f $diff_file
	
	if [ ! -z "$lint" ]; then
		printf "Fix $f ? (y/n): "
		read ans
		if [ "$ans" == "y" ]; then
			git-clang-format --patch --force $target -- $f
		fi
	fi
done
```

* Make sure your code compiles with extra error checking:   
```
./configure CC=gcc <extra-flags ...>
make CFLAGS="-Wextra -Wall -Werror -pedantic"
```

* Make sure your code is tested and existing and new unit-tests run and pass:   
```
make CFLAGS="-Wextra -Wall -Werror -pedantic" check
```

* Make sure your code is documented the documentation builds without errors:   
```
make html
```
