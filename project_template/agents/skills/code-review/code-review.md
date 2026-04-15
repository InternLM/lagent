---
name: code-review
description: Code review best practices and checklist
---

# Code Review Skill

When reviewing code, check the following:

## Security
- No hardcoded secrets or API keys
- Input validation on all user-facing endpoints
- SQL injection / XSS prevention

## Quality
- Functions are < 50 lines
- No duplicated logic (DRY)
- Meaningful variable names
- Error handling for all I/O operations

## Testing
- New functions have corresponding tests
- Edge cases are covered
- Test names describe the expected behavior
