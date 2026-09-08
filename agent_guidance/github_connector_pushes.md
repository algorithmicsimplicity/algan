# Publishing with the ChatGPT GitHub connector

Read this when a ChatGPT/Codex session needs to publish a multi-file change through the GitHub connector rather than a normal authenticated `git push`.

## Preferred path: Git data objects

For a multi-file change, especially one containing binary files, use GitHub's low-level Git data operations. Do not stop after creating the branch: `create_branch` only creates or moves a ref to the chosen base commit; it does not publish the working tree.

1. Resolve and record the exact base commit SHA and its tree SHA. Work against that immutable commit, not a moving branch name.
2. Create the destination branch at that base commit if it does not already exist.
3. Build and validate the complete final tree locally. Run `git diff --check` and the task's required tests before publication.
4. Upload every changed file with `create_blob`:
   - UTF-8 source/docs/config: `encoding="utf-8"`.
   - Binary data: base64-encode the raw bytes and use `encoding="base64"`.
   Record the returned blob SHA for every file. Do not try to read a binary blob back through UTF-8-only file endpoints.
5. Call `create_tree` with the base tree SHA and one entry per changed path. Use the correct Git mode (normally `100644`; preserve executable modes when applicable), `type="blob"`, and the uploaded blob SHA. Represent deletions explicitly rather than omitting them.
6. Call `create_commit` with the new tree SHA and the exact base commit as its parent. This creates the commit object but still does not move the branch.
7. Publish by calling `update_ref` on the destination branch with the new commit SHA. Use `force=false` for the normal fast-forward path. Only force-update when the user explicitly wants history rewritten and the consequences have been checked.
8. Verify publication with `compare_commits(base, head)` and/or `fetch_commit`. Confirm the expected changed filenames, no helper files, the intended parent commit, and the target branch head.
9. Open or update the PR only after the branch is verified. Do not create a scheduled CI-monitoring task unless the user explicitly asks for one.

For a single existing UTF-8 file, `fetch_file` + `update_file` is simpler. Fetch the current blob SHA first. For a single new UTF-8 file, `create_file` is sufficient. These contents-API helpers are not the preferred route for atomic multi-file changes and are unsuitable for binary content.

## Common failure modes

- **Branch exists but has no changes:** the session called `create_branch` but never completed `create_commit` + `update_ref`.
- **Binary file fails with UnicodeDecodeError:** a UTF-8 fetch/update path was used for binary data. Upload its bytes with `create_blob(..., encoding="base64")` instead.
- **Partial multi-file publication:** sequential `update_file` calls created one commit per file or stopped midway. Prefer one tree and one commit.
- **Wrong parent/base:** a temporary helper branch was used as the parent. Build the production commit directly on the recorded base commit so helper workflows, patch fragments, or export scripts cannot enter the PR history.
- **Unverified object creation:** `create_blob`, `create_tree`, and `create_commit` can produce unreferenced objects. They do not affect a branch until `update_ref`; verify the tree before moving the ref.

## Fallback: one-off GitHub Actions publisher

Use a temporary Actions workflow only when direct Git data upload is genuinely impractical. Give the job `contents: write`, copy any patch payload out of the helper branch, check out the exact production base commit, apply and validate the patch there, create one clean implementation commit, then push that commit to the destination branch. Never make the production commit a child of the temporary helper commits. Remove helper files from the production tree and reset or delete the temporary branch afterward when the connector permits it.

A one-off publisher is a transport workaround, not validation. The final branch must still be inspected with GitHub's compare/commit APIs.
