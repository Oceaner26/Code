#!/usr/bin/env python3
"""Verify source fidelity after comment translation and removal of unused scripts.

Compares exact file hashes and executable ASTs against fingerprints captured
from the supplied Code(8).zip. Documentation strings are excluded from the
AST comparison because they were translated into English intentionally.
This does not validate scientific results or reproduce historical experiments.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'provenance' / 'source_integrity.json'


def file_hash(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def executable_ast_hash(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body[0] = ast.Pass()
    return file_hash(ast.dump(tree, include_attributes=False).encode('utf-8'))


def main() -> int:
    spec = json.loads(MANIFEST.read_text(encoding='utf-8'))
    errors = []
    for rel, info in sorted(spec['files'].items()):
        path = ROOT / rel
        if not path.is_file():
            errors.append(f'MISSING {rel}')
            continue
        raw = path.read_bytes()
        if file_hash(raw) != info['release_sha256']:
            errors.append(f'FILE HASH CHANGED {rel}')
        if rel.endswith('.py'):
            source = raw.decode('utf-8-sig')
            try:
                ast_digest = executable_ast_hash(source)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f'PYTHON PARSE FAILED {rel}: {exc}')
                continue
            if ast_digest != info['executable_ast_sha256']:
                errors.append(f'EXECUTABLE CODE CHANGED {rel}')
            try:
                comments = [tok for tok in tokenize.generate_tokens(io.StringIO(source).readline) if tok.type == tokenize.COMMENT]
                if any(re.search(r'[\u3400-\u9fff]', tok.string) for tok in comments):
                    errors.append(f'NON-ENGLISH COMMENT {rel}')
            except tokenize.TokenError as exc:
                errors.append(f'TOKENIZATION FAILED {rel}: {exc}')
    print(f'Preserved source-derived files checked: {len(spec["files"])}')
    print(f'Python modules with executable AST signatures: {sum(s.endswith(".py") for s in spec["files"])}')
    if errors:
        for error in errors:
            print('FAIL:', error)
        return 1
    print('PASS: file hashes, executable Python ASTs, and comment language verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
