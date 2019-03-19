Contributing to The Project
===========================

The following is a set of guidelines for contributing to this project. These
are guidelines, not rules, so use your best judgment and feel free to propose
changes. For information about the overall design of the library and general
coding guidelines, see the `DESIGN.markdown` file.

## Contribution Workflow

We basically follow the [Gitlab Flow](https://docs.gitlab.com/ee/workflow/gitlab_flow.html)
model:
 - `master` is the primary branch for up-to-date development
 - any new feature/development should go into a `feature-branch` based of
   `master` and merge into `master` after.
 - we follow a `semi-linear history with merge commits` strategy: each merge
   request should include clean commits, but only the HEAD and merge commit
   need to pass CI.
 - if the feature branch is not too complex, a single commit, even from a
   squash is okay. If the feature branch includes too many changes, each major
   change should appear in its own commit.
 - to help with in-development work, CI will not activate on branches with
   names started with wip/
## Commit Messages Styleguide

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

## Signoff on Contributions:

The project uses the [Developer Certificate of
Origin](https://developercertificate.org/) for copyright and license
management. We ask that you sign-off all of your commits as certification that
you have the rights to submit this work under the license of the project (in
the `LICENSE` file) and that you agree to the DCO.

To signoff commits, use: `git commit --signoff`.
To signoff a branch after the fact: `git rebase --signoff`
