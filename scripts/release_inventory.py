#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / ".github" / "release-packages.json"
JS_WORKSPACE_ROOT = REPO_ROOT / "javascript-sdks"


def load_inventory() -> list[dict]:
    data = json.loads(INVENTORY_PATH.read_text())
    return data["packages"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_toml(path: Path) -> dict:
    text = path.read_text()
    return parse_toml_text(text)


def parse_toml_text(text: str) -> dict:
    if tomllib is not None:
        return tomllib.loads(text)
    return parse_minimal_toml(text)


def parse_minimal_toml(text: str) -> dict:
    data: dict = {}
    section: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].split(".")
            target = data
            for part in section:
                target = target.setdefault(part, {})
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        string_match = re.fullmatch(r'"(.*)"', value)
        if not string_match:
            continue

        target = data
        for part in section:
            target = target.setdefault(part, {})
        target[key] = string_match.group(1)

    return data


def manifest_path(entry: dict) -> Path:
    filename = "package.json" if entry["ecosystem"] == "javascript" else "pyproject.toml"
    return REPO_ROOT / entry["path"] / filename


def load_manifest(entry: dict) -> dict:
    path = manifest_path(entry)
    if entry["ecosystem"] == "javascript":
        return read_json(path)
    return read_toml(path)


def manifest_name(entry: dict, manifest: dict) -> str:
    if entry["ecosystem"] == "javascript":
        return manifest["name"]
    return manifest["tool"]["poetry"]["name"]


def manifest_version(entry: dict, manifest: dict) -> str:
    if entry["ecosystem"] == "javascript":
        return manifest["version"]
    return manifest["tool"]["poetry"]["version"]


def python_import_name(manifest: dict) -> str | None:
    poetry = manifest.get("tool", {}).get("poetry", {})
    packages = poetry.get("packages", [])
    if isinstance(packages, list) and packages:
        first = packages[0]
        if isinstance(first, dict):
            include = first.get("include")
            if isinstance(include, str) and include:
                return include
    return None


def javascript_bin_name(manifest: dict) -> str | None:
    bin_field = manifest.get("bin")
    if isinstance(bin_field, str):
        return manifest.get("name")
    if isinstance(bin_field, dict) and bin_field:
        return next(iter(bin_field))
    return None


def has_js_script(manifest: dict, script_name: str) -> bool:
    return script_name in manifest.get("scripts", {})


def slug_for(entry: dict) -> str:
    return entry["path"].split("/")[-1].replace(".", "-")


def git_stdout(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def git_stdout_or_none(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def changed_files(base: str, head: str) -> set[str]:
    if not base or set(base) == {"0"}:
        return {
            str(path.relative_to(REPO_ROOT))
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        }
    output = git_stdout_or_none("diff", "--name-only", f"{base}...{head}")
    if output is None:
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def is_non_substantive_path(path: str) -> bool:
    pure = PurePosixPath(path)

    if pure.name == ".DS_Store":
        return True
    if pure.suffix == ".pyc":
        return True
    if "__pycache__" in pure.parts:
        return True
    if ".pytest_cache" in pure.parts or ".mypy_cache" in pure.parts:
        return True
    if pure.name in {".env", ".env.backup"}:
        return True
    if pure.parts[-2:] == (".yarn", "install-state.gz"):
        return True

    return False


def package_changed(entry: dict, files: set[str]) -> bool:
    prefix = f"{entry['path']}/"
    return any(
        (path == entry["path"] or path.startswith(prefix)) and not is_non_substantive_path(path)
        for path in files
    )


def version_from_git_ref(entry: dict, ref: str) -> str | None:
    if not ref:
        return None
    repo_path = f"{entry['path']}/{'package.json' if entry['ecosystem'] == 'javascript' else 'pyproject.toml'}"
    content = git_stdout_or_none("show", f"{ref}:{repo_path}")
    if content is None:
        return None
    if entry["ecosystem"] == "javascript":
        manifest = json.loads(content)
        return manifest["version"]
    manifest = parse_toml_text(content)
    return manifest["tool"]["poetry"]["version"]


def workspace_paths(entries: list[dict]) -> list[str]:
    paths = []
    for entry in entries:
        if entry["ecosystem"] != "javascript":
            continue
        paths.append(entry["path"].split("/", 1)[1])
    return sorted(paths)


def js_internal_dependency_names(entries: list[dict]) -> dict[str, set[str]]:
    js_entries = {entry["name"]: entry for entry in entries if entry["ecosystem"] == "javascript"}
    manifests = {name: load_manifest(entry) for name, entry in js_entries.items()}
    graph: dict[str, set[str]] = {}

    for name, manifest in manifests.items():
        internal: set[str] = set()
        for section in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
            for dependency_name in manifest.get(section, {}):
                if dependency_name != name and dependency_name in js_entries:
                    internal.add(dependency_name)
        graph[name] = internal

    return graph


def python_internal_dependency_names(entries: list[dict]) -> dict[str, set[str]]:
    py_entries = {entry["name"]: entry for entry in entries if entry["ecosystem"] == "python"}
    manifests = {name: load_manifest(entry) for name, entry in py_entries.items()}
    graph: dict[str, set[str]] = {}

    for name, manifest in manifests.items():
        internal: set[str] = set()

        poetry_dependencies = manifest.get("tool", {}).get("poetry", {}).get("dependencies", {})
        for dependency_name in poetry_dependencies:
            if dependency_name != "python" and dependency_name in py_entries:
                internal.add(dependency_name)

        project_dependencies = manifest.get("project", {}).get("dependencies", [])
        for dependency_spec in project_dependencies:
            match = re.match(r"^\s*([A-Za-z0-9_.-]+)", dependency_spec)
            if not match:
                continue
            dependency_name = match.group(1)
            if dependency_name in py_entries:
                internal.add(dependency_name)

        graph[name] = internal

    return graph


def internal_dependency_names(entries: list[dict], ecosystem: str) -> dict[str, set[str]]:
    if ecosystem == "javascript":
        return js_internal_dependency_names(entries)
    if ecosystem == "python":
        return python_internal_dependency_names(entries)
    raise ValueError(f"unsupported ecosystem for dependency graph: {ecosystem}")


def expand_with_dependents(entries: list[dict], selected: list[dict], ecosystem: str) -> list[dict]:
    if not selected:
        return []

    graph = internal_dependency_names(entries, ecosystem)
    reverse_graph: dict[str, set[str]] = {entry["name"]: set() for entry in entries if entry["ecosystem"] == ecosystem}

    for package_name, dependency_names in graph.items():
        for dependency_name in dependency_names:
            reverse_graph.setdefault(dependency_name, set()).add(package_name)

    selected_names = {entry["name"] for entry in selected}
    queue = list(selected_names)

    while queue:
        current = queue.pop()
        for dependent_name in reverse_graph.get(current, set()):
            if dependent_name in selected_names:
                continue
            selected_names.add(dependent_name)
            queue.append(dependent_name)

    ordered_entries = [entry for entry in entries if entry["ecosystem"] == ecosystem and entry["name"] in selected_names]
    return ordered_entries


def ecosystem_shared_change(entry: dict, files: set[str]) -> bool:
    if entry["ecosystem"] == "javascript":
        shared_paths = {
            "javascript-sdks/package.json",
            "javascript-sdks/yarn.lock",
        }
        if any(path in shared_paths for path in files):
            return True
        if any(path.startswith("javascript-sdks/.yarn/") for path in files):
            return True
    if entry["ecosystem"] == "python":
        shared_paths = {
            "python-sdks/pyproject.toml",
            "python-sdks/poetry.lock",
            "python-sdks/pytest.ini",
            "python-sdks/setup.cfg",
            "python-sdks/tox.ini",
            "python-sdks/requirements.txt",
            "python-sdks/requirements-dev.txt",
            "python-sdks/constraints.txt",
        }
        if any(path in shared_paths for path in files):
            return True
        if any(re.fullmatch(r"python-sdks/requirements(?:[-_.][^/]+)?\.txt", path) for path in files):
            return True
        if any(re.fullmatch(r"python-sdks/[^/]+\.(toml|ini|cfg)", path) for path in files):
            return True
    return False


def javascript_build_order(entries: list[dict], package_name: str) -> list[dict]:
    js_entries = {entry["name"]: entry for entry in entries if entry["ecosystem"] == "javascript"}
    if package_name not in js_entries:
        raise KeyError(package_name)

    graph = js_internal_dependency_names(entries)
    ordered_names: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"cyclic javascript dependency detected at {name}")

        visiting.add(name)
        for dependency_name in sorted(graph.get(name, set())):
            visit(dependency_name)
        visiting.remove(name)
        visited.add(name)
        ordered_names.append(name)

    visit(package_name)
    return [js_entries[name] for name in ordered_names]


def validate(entries: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_names: set[tuple[str, str]] = set()

    js_root_manifest = read_json(JS_WORKSPACE_ROOT / "package.json")
    actual_workspaces = set(js_root_manifest.get("workspaces", []))
    expected_workspaces = set(workspace_paths(entries))
    missing_workspaces = sorted(expected_workspaces - actual_workspaces)
    if missing_workspaces:
        errors.append(
            "javascript workspace list is missing release inventory packages: "
            f"{missing_workspaces}"
        )

    for entry in entries:
        key = (entry["ecosystem"], entry["name"])
        if key in seen_names:
            errors.append(f"duplicate inventory entry for {entry['ecosystem']} package {entry['name']}")
            continue
        seen_names.add(key)

        if "/legacy/" in entry["path"]:
            errors.append(f"{entry['name']} is under a legacy path and must not be release-managed")
            continue

        path = manifest_path(entry)
        if not path.exists():
            errors.append(f"missing manifest for {entry['name']} at {path.relative_to(REPO_ROOT)}")
            continue

        manifest = load_manifest(entry)
        if manifest_name(entry, manifest) != entry["name"]:
            errors.append(
                f"inventory name mismatch for {entry['path']}: "
                f"expected {entry['name']}, got {manifest_name(entry, manifest)}"
            )

        if entry["ecosystem"] == "javascript":
            if manifest.get("private"):
                errors.append(f"{entry['name']} is marked private but included in release inventory")
            publish_access = manifest.get("publishConfig", {}).get("access")
            if entry["registry"] == "npm" and publish_access != "public":
                errors.append(f"{entry['name']} is missing publishConfig.access=public")
            # `npm publish` does not rewrite the `workspace:` protocol (only
            # yarn berry / pnpm do), so any workspace: range leaks verbatim into
            # the published tarball and makes the package uninstallable from npm.
            for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                for dep_name, dep_range in (manifest.get(section) or {}).items():
                    if isinstance(dep_range, str) and dep_range.startswith("workspace:"):
                        errors.append(
                            f"{entry['name']} declares {section}.{dep_name} as "
                            f"'{dep_range}'; replace the workspace: range with a "
                            f"concrete version before publishing"
                        )
        else:
            project_version = manifest.get("project", {}).get("version")
            poetry_version = manifest["tool"]["poetry"]["version"]
            if project_version and project_version != poetry_version:
                errors.append(
                    f"{entry['name']} has conflicting [project] and [tool.poetry] versions: "
                    f"{project_version} != {poetry_version}"
                )

    return errors


def build_record(entry: dict, manifest: dict) -> dict:
    record = {
        "ecosystem": entry["ecosystem"],
        "name": entry["name"],
        "path": entry["path"],
        "registry": entry["registry"],
        "slug": slug_for(entry),
        "version": manifest_version(entry, manifest),
    }
    if entry["ecosystem"] == "javascript":
        record["has_build"] = has_js_script(manifest, "build")
        record["has_test"] = has_js_script(manifest, "test")
        record["bin_name"] = javascript_bin_name(manifest)
    else:
        record["import_name"] = python_import_name(manifest)
    return record


def filtered_entries(
    entries: list[dict],
    ecosystem: str,
    base: str | None,
    head: str | None,
    require_version_change: bool,
    include_dependents: bool = False,
) -> list[dict]:
    selected = [entry for entry in entries if ecosystem == "all" or entry["ecosystem"] == ecosystem]
    manifests = {entry["name"]: load_manifest(entry) for entry in selected}

    if base and head:
        files = changed_files(base, head)
        selected = [
            entry
            for entry in selected
            if package_changed(entry, files) or ecosystem_shared_change(entry, files)
        ]
        if include_dependents and ecosystem in {"javascript", "python"}:
            selected = expand_with_dependents(entries, selected, ecosystem)

    if require_version_change:
        filtered: list[dict] = []
        for entry in selected:
            current_version = manifest_version(entry, manifests[entry["name"]])
            previous_version = version_from_git_ref(entry, base or "")
            if previous_version != current_version:
                filtered.append(entry)
        selected = filtered

    return [build_record(entry, manifests[entry["name"]]) for entry in selected]


def entries_missing_version_bump(
    entries: list[dict],
    ecosystem: str,
    base: str | None,
    head: str | None,
) -> list[dict]:
    selected = [entry for entry in entries if ecosystem == "all" or entry["ecosystem"] == ecosystem]
    manifests = {entry["name"]: load_manifest(entry) for entry in selected}

    if base and head:
        files = changed_files(base, head)
        selected = [entry for entry in selected if package_changed(entry, files)]
    else:
        selected = []

    missing: list[dict] = []
    for entry in selected:
        current_version = manifest_version(entry, manifests[entry["name"]])
        previous_version = version_from_git_ref(entry, base or "")
        if previous_version == current_version:
            missing.append(build_record(entry, manifests[entry["name"]]))

    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecosystem", choices=["all", "javascript", "python"], default="all")
    parser.add_argument("--changed-from")
    parser.add_argument("--changed-to")
    parser.add_argument("--version-changed", action="store_true")
    parser.add_argument("--missing-version-bump", action="store_true")
    parser.add_argument("--include-dependents", action="store_true")
    parser.add_argument("--build-order-for")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--format", choices=["json", "github-matrix", "paths", "names"], default="json")
    args = parser.parse_args()

    entries = load_inventory()

    if args.validate:
        errors = validate(entries)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("release metadata validation passed")
        return 0

    if args.build_order_for:
        try:
            selected_entries = javascript_build_order(entries, args.build_order_for)
        except KeyError:
            print(f"unknown javascript package: {args.build_order_for}", file=sys.stderr)
            return 1
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1

        if args.format == "paths":
            for entry in selected_entries:
                print(entry["path"])
            return 0
        if args.format == "names":
            for entry in selected_entries:
                print(entry["name"])
            return 0

        selected = [build_record(entry, load_manifest(entry)) for entry in selected_entries]
        if args.count:
            print(len(selected))
            return 0
        if args.format == "github-matrix":
            print(json.dumps(selected, separators=(",", ":")))
            return 0
        print(json.dumps(selected, indent=2))
        return 0

    if args.version_changed and args.missing_version_bump:
        print(
            "--version-changed and --missing-version-bump are mutually exclusive",
            file=sys.stderr,
        )
        return 1

    if args.missing_version_bump:
        selected = entries_missing_version_bump(
            entries,
            ecosystem=args.ecosystem,
            base=args.changed_from,
            head=args.changed_to,
        )
        if args.count:
            print(len(selected))
            return 0
        if args.format == "github-matrix":
            print(json.dumps(selected, separators=(",", ":")))
            return 0
        if args.format == "names":
            for entry in selected:
                print(entry["name"])
            return 0
        if args.format == "paths":
            for entry in selected:
                print(entry["path"])
            return 0
        print(json.dumps(selected, indent=2))
        return 0

    selected = filtered_entries(
        entries,
        ecosystem=args.ecosystem,
        base=args.changed_from,
        head=args.changed_to,
        require_version_change=args.version_changed,
        include_dependents=args.include_dependents,
    )

    if args.count:
        print(len(selected))
        return 0

    if args.format == "github-matrix":
        print(json.dumps(selected, separators=(",", ":")))
        return 0

    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
