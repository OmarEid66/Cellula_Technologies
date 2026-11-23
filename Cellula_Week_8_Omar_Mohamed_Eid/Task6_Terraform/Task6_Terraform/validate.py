"""Static checks for the Terraform configuration.

`terraform init/validate/plan` need the AWS provider download, which this
machine cannot install. This script covers what those commands would catch
without network access: every file parses, every variable that is referenced
is declared, module inputs and outputs line up, and terraform.tfvars.example
supplies the variables it overrides.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import hcl2

ROOT = Path(__file__).resolve().parent
MODULES = ROOT / "modules"

REF = re.compile(r"\bvar\.([A-Za-z0-9_]+)")
OUT = re.compile(r"\bmodule\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")


def parse(path: Path) -> dict:
    with path.open() as fh:
        return hcl2.load(fh)


def clean(key: str) -> str:
    return key.strip('"')


def named_blocks(doc: dict, kind: str):
    for block in doc.get(kind, []):
        for key, value in block.items():
            if key != "__is_block__":
                yield clean(key), value


def declared(doc: dict, kind: str) -> set[str]:
    return {name for name, _ in named_blocks(doc, kind)}


def body_text(doc: dict) -> str:
    text = []
    for kind in ("resource", "data", "module", "locals"):
        for _, value in named_blocks(doc, kind):
            text.append(repr(value))
    return "\n".join(text)


def main() -> int:
    errors: list[str] = []
    files = sorted(ROOT.glob("*.tf")) + sorted(MODULES.glob("*/*.tf"))
    docs = {p: parse(p) for p in files}

    for name in ("providers.tf", "main.tf", "variables.tf", "outputs.tf"):
        if not (ROOT / name).is_file():
            errors.append(f"missing required file {name}")

    root_docs = [docs[p] for p in files if p.parent == ROOT]
    root_vars = set().union(*(declared(d, "variable") for d in root_docs))
    root_outs = set().union(*(declared(d, "output") for d in root_docs))
    root_text = "\n".join(body_text(d) for d in root_docs)

    for var in sorted(set(REF.findall(root_text)) - root_vars):
        errors.append(f"root references undeclared var.{var}")

    root_modules = {
        name: cfg for d in root_docs for name, cfg in named_blocks(d, "module")
    }
    for mod_name, out in sorted(set(OUT.findall(root_text))):
        if mod_name not in root_modules:
            errors.append(f"references module.{mod_name} which is not declared")
            continue
        outputs_file = MODULES / mod_name / "outputs.tf"
        if not outputs_file.is_file() or out not in declared(parse(outputs_file), "output"):
            errors.append(f"module {mod_name} has no output {out}")

    for mod_name, cfg in root_modules.items():
        source = cfg.get("source", "").strip('"')
        if not source.startswith("./"):
            continue
        target = (ROOT / source).resolve()
        if not target.is_dir():
            errors.append(f"module {mod_name}: source {source} does not exist")
            continue
        mod_docs = [parse(p) for p in sorted(target.glob("*.tf"))]
        mod_vars = {}
        for d in mod_docs:
            for name, body in named_blocks(d, "variable"):
                mod_vars[name] = "default" in body
        supplied = {clean(k) for k in cfg if k not in {"__is_block__", "source", "version"}}
        for key in sorted(supplied - set(mod_vars)):
            errors.append(f"module {mod_name}: unknown input {key}")
        for key, has_default in sorted(mod_vars.items()):
            if key not in supplied and not has_default:
                errors.append(f"module {mod_name}: required input {key} is not passed")
        mod_text = "\n".join(body_text(d) for d in mod_docs)
        for var in sorted(set(REF.findall(mod_text)) - set(mod_vars)):
            errors.append(f"module {mod_name}: uses var.{var} but does not declare it")

    tfvars = {clean(k) for k in parse(ROOT / "terraform.tfvars.example")}
    for key in sorted(tfvars - root_vars):
        errors.append(f"terraform.tfvars.example sets unknown variable {key}")

    blocks = sum(1 for d in docs.values() for k in ("resource", "data", "module") for _ in named_blocks(d, k))
    resources = sum(1 for d in docs.values() for _ in named_blocks(d, "resource"))
    print(f"parsed {len(files)} files, {len(root_vars)} root variables, {blocks} blocks, {resources} resources")

    if errors:
        print("\nFAILED:")
        for err in errors:
            print(" -", err)
        return 1

    print("OK: syntax parses and cross-references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
