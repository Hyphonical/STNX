# Copilot Instructions for Rust Development

## Agent Operating Rules

Read the codebase before making changes. Build a real understanding of the relevant source files, surrounding modules, existing patterns, and project structure before editing anything.

Find and read `PLAN.md` before starting work whenever it exists. If there are multiple plan files, use the one most relevant to the area you are modifying. Do not begin implementation until you have checked whether a plan already exists.

Do not start work if you only have partial context. Gather enough information first using the available tools. If requirements, scope, constraints, or expected behavior are unclear, ask questions using tools only. Do not ask questions in chat. Do not guess.

### Tool Usage and Failure Handling

Use the available tools as the default way to gather information, inspect the codebase, and ask for missing context.

If a tool fails, do not use workarounds to compensate for the failure. If command output is unavailable, do not redirect output into a file, do not create a temporary script to dump the output somewhere else, and do not invent alternate extraction methods. Report the failure directly.

If an edit fails, report the failure directly. Do not pipe output into files. Do not create Python, PowerShell, or other helper snippets just to work around the failed edit. Tell the user what failed.

If you hit the same error on the same file more than once, use web search instead of repeating the same failing approach. Do not keep retrying the same broken path.

Do not create random helper files or snippets without approval. This includes temporary Python files, PowerShell scripts, ad hoc test harnesses, one-off migration scripts, and similar files. Ask first.

### Shell Environment

The shell environment is PowerShell. Use PowerShell syntax and conventions only.

Do not use bash commands, bash syntax, or bash-specific idioms.

Do not run random git commands. Only use git commands when they are clearly necessary for the requested task.

### File Safety and Approval Boundaries

Do not edit sensitive project files without asking first. This includes files such as:

- `requirements.txt`
- `Cargo.toml`
- `Cargo.lock`
- lockfiles
- dependency manifests
- CI configuration
- environment configuration
- project-wide toolchain or build configuration

Do not assume approval for risky or wide-impact edits. Ask first.

## Core Principles

You are working on a Rust codebase that must maintain high quality standards. Every change should improve code quality, readability, and maintainability. Write code as if it will be reviewed by experienced Rust developers and maintained by a team over several years.

## Code Quality Standards

### Compiler and Linter Compliance

Resolve all compiler errors and warnings before considering work complete. Run `cargo check` to verify compilation succeeds without errors. Execute `cargo clippy` with default settings and address every warning it produces. When Clippy suggests a change, implement the recommended fix rather than suppressing the warning unless you have a compelling technical reason documented in a comment. Run `cargo fmt` to ensure consistent code formatting across the entire codebase.

### Code Cleanliness

Remove all unused imports, functions, variables, type definitions, and dependencies. If code exists in the codebase but serves no current purpose, delete it rather than commenting it out. Trust version control to preserve history if the code becomes needed again in the future. Eliminate duplicate code by extracting shared logic into well-named functions or modules. When you identify repeated patterns, refactor them into reusable abstractions.

### Rust Idioms and Best Practices

Write idiomatic Rust that follows community conventions and leverages the language's strengths. Prefer iterator methods over explicit loops when working with collections. Use pattern matching exhaustively rather than cascading if statements. Leverage the type system to encode invariants and prevent invalid states. Follow ownership and borrowing patterns correctly, avoiding unnecessary cloning while maintaining clarity. Use references appropriately and understand when to use mutable versus immutable borrows.

## Error Handling

Handle errors explicitly and thoughtfully throughout the codebase. Use Result types for operations that can fail and propagate errors with the question mark operator. Avoid calling `unwrap()` or `expect()` in production code paths unless you can prove the operation cannot fail and document why. For cases where failure truly represents a programming error rather than a runtime condition, consider using assertions or dedicated panic macros with clear messages.

Create custom error types when appropriate to provide meaningful context about failures. Implement proper error messages that help users and developers understand what went wrong and how to address it. Consider using established error handling libraries like `thiserror` or `anyhow` when they simplify error management without adding unnecessary complexity.

## Testing Requirements

Write tests that verify correctness and prevent regressions. Create unit tests for individual functions and modules, placing them in a tests submodule within each source file. Write integration tests in the tests directory for end-to-end functionality. Test both success paths and error conditions to ensure proper handling of edge cases and invalid input.

Mock external dependencies like network calls or file system operations to make tests deterministic and fast. Use property-based testing with libraries like `proptest` or `quickcheck` when appropriate to verify behavior across a wide range of inputs. Ensure tests are independent and can run in any order without affecting each other.

Write tests that are clear and maintainable. Each test should verify one specific behavior with a descriptive name that explains what it validates. Include assertions with meaningful messages that make failures easy to diagnose. When tests fail, the error message should immediately indicate what went wrong.

## Documentation

Document public APIs with doc comments that explain purpose, parameters, return values, and any invariants or preconditions. Write documentation that helps other developers use your code correctly without reading the implementation. Include usage examples in doc comments when they clarify intent or demonstrate common patterns.

Use inline comments sparingly and only to explain why code does something non-obvious, not what it does. Well-named functions and variables should make the code self-documenting in most cases. When you encountered unclear code, refactor it for clarity rather than adding explanatory comments.

## Code Organization and Architecture

Maintain clear separation of concerns across modules. Group related functionality together and minimize dependencies between different parts of the system. Keep modules focused on a single responsibility and avoid creating large files that mix multiple concerns.

Use meaningful names for functions, variables, types, and modules that clearly convey their purpose. Avoid abbreviations unless they are widely recognized in the domain. Choose names that make code read naturally and reduce the need for explanatory comments.

Structure code to separate pure business logic from side effects like network IO, file system access, or UI rendering. This separation makes code easier to test, understand, and modify. When possible, pass dependencies explicitly through function parameters or constructors rather than accessing global state.

## Performance and Efficiency

Write clear, correct code first and optimize only when profiling indicates a performance bottleneck. Avoid premature optimization that complicates code without measurable benefit. When optimization is necessary, measure the impact before and after changes to verify improvement.

Be mindful of common performance pitfalls in Rust such as unnecessary cloning, allocation in hot paths, or inefficient algorithms. Use appropriate data structures for the access patterns in your code. Consider using crates like `parking_lot` for improved lock performance or `smallvec` for stack-allocated collections when profiling demonstrates the need.

## Communication and Questions

Ask clarifying questions when requirements are ambiguous or when multiple implementation approaches exist with different tradeoffs. Use the available tools to inquire about design decisions, gather context, or request feedback before making significant architectural changes.

When asking questions, provide relevant context about what you are trying to accomplish and what specific aspect needs clarification. Present options with their respective advantages and disadvantages when seeking guidance on technical decisions. Be specific rather than asking overly broad questions. Ask questions using tools only; do not ask questions in chat.

Do not proceed with implementation if you only understand part of the task. Gather enough context to work with confidence before making changes.

## Dependencies and External Crates

Evaluate dependencies carefully before adding them to the project. Choose well-maintained crates with active communities, good documentation, and appropriate licenses. Prefer crates that follow semantic versioning and have stable APIs. Avoid adding dependencies for trivial functionality that could be easily implemented directly.

Keep dependencies up to date with security patches while testing updates before deploying them. Review dependency trees to understand transitive dependencies and watch for bloat or conflicting requirements.

Do not edit dependency files like `Cargo.toml` or lockfiles without explicit approval.

## Incremental Development

Make changes incrementally with focused commits that address one concern at a time. Each commit should leave the codebase in a working state with passing tests. Write clear commit messages that explain what changed and why, helping future maintainers understand the evolution of the codebase.

Break large features into smaller reviewable pieces when possible. This approach makes it easier to verify correctness, simplifies debugging when issues arise, and allows for iterative feedback during development.

## Final Verification

Before considering any work complete:

- Confirm the relevant plan (`PLAN.md`) was checked
- Confirm the necessary context was gathered first
- Confirm no forbidden workaround was used
- Run the full test suite to ensure nothing broke
- Execute Clippy one final time to catch any warnings introduced during implementation
- Format the code with rustfmt for consistency
- Review your changes critically as if you were conducting a code review for a colleague

### Summary Requirements

Report all warnings and errors encountered during the session in your final summary, even if they were not caused by your edits and even if they were outside the exact files changed. This includes any tool failures, command failures, or edit failures encountered during the session.