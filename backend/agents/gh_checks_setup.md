# GitHub Required Checks Setup

To enforce the quality gates before merging pull requests, you need to configure branch protection rules for your `main` branch.

## Steps

1.  **Navigate to Branch Protection Rules**:
    Go to your repository on GitHub, then click `Settings` > `Branches`. Under "Branch protection rules", click `Add rule`.

2.  **Protect the `main` Branch**:
    - In "Branch name pattern", enter `main`.

3.  **Require Status Checks to Pass Before Merging**:
    - Check the box for `Require status checks to pass before merging`.
    - Check `Require branches to be up to date before merging`. This ensures PRs are tested against the latest `main`.

4.  **Add Required Status Checks**:
    - In the search box under "Status checks that are required", search for and select the jobs from your `quality.yml` workflow:
      - `lint`
      - `test`
      - `deps`

5.  **Save Changes**:
    - Click `Create` or `Save changes` at the bottom of the page.

Now, all pull requests targeting the `main` branch will be blocked from merging until the `lint`, `test`, and `deps` checks have passed successfully.
