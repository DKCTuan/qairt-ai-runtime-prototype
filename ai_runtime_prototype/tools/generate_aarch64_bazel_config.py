#!/usr/bin/env python3
"""Create a portable ARM64-only TensorFlow 2.15 Bazel toolchain config.

TensorFlow's generated ``local_config_embedded_arm`` writes Bazel-cache paths
into cc_config.bzl.  Such a directory cannot be copied to another host.  This
helper instead writes absolute paths to the selected ARM64 toolchain and
creates the minimal ARM64-only repository required by --config=elinux_aarch64.
It supports the standard ARM GNU compiler and vendor compilers such as
OpenWrt's ``aarch64-openwrt-linux-musl-gcc``.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def compiler_version(compiler: Path) -> str:
    result = subprocess.run(
        [str(compiler), "-dumpfullversion", "-dumpversion"],
        check=True, capture_output=True, text=True)
    version = result.stdout.strip().splitlines()[0]
    if not version or any(part and not part.isdigit() for part in version.split(".")):
        fail(f"cannot determine GCC version from {compiler}: {version!r}")
    return version


def compiler_target(compiler: Path) -> str:
    """Derive the target triplet from an aarch64 ``*-gcc`` executable."""
    name = compiler.name
    if not name.endswith("-gcc"):
        fail(f"compiler name must end in -gcc: {compiler}")
    target = name[:-4]
    if not target.startswith("aarch64-"):
        fail(f"compiler is not an aarch64 cross compiler: {compiler}")
    return target


def write_build(destination: Path, *, include_compiler_wrapper: bool) -> None:
    wrapper_target = (
        'filegroup(name = "compiler_wrapper", srcs = ["compiler_wrapper.sh"])\n\n'
        if include_compiler_wrapper else '')
    wrapper_src = '":compiler_wrapper", ' if include_compiler_wrapper else ''
    destination.write_text(
        'load(":cc_config.bzl", "cc_toolchain_config")\n\n'
        'package(default_visibility = ["//visibility:public"])\n\n'
        'cc_toolchain_suite(\n'
        '    name = "toolchain",\n'
        '    toolchains = {"aarch64": ":cc-compiler-aarch64"},\n'
        ')\n\n'
        'filegroup(name = "empty", srcs = [])\n\n'
        + wrapper_target +
        'filegroup(\n'
        '    name = "aarch64_toolchain_all_files",\n'
        f'    srcs = [{wrapper_src}"@aarch64_linux_toolchain//:compiler_pieces"],\n'
        ')\n\n'
        'cc_toolchain_config(name = "aarch64_toolchain_config", cpu = "aarch64")\n\n'
        'cc_toolchain(\n'
        '    name = "cc-compiler-aarch64",\n'
        '    all_files = ":aarch64_toolchain_all_files",\n'
        '    compiler_files = ":aarch64_toolchain_all_files",\n'
        '    dwp_files = ":empty",\n'
        '    linker_files = ":aarch64_toolchain_all_files",\n'
        '    objcopy_files = ":aarch64_toolchain_all_files",\n'
        '    strip_files = ":aarch64_toolchain_all_files",\n'
        '    supports_param_files = 1,\n'
        '    toolchain_config = ":aarch64_toolchain_config",\n'
        ')\n', encoding="utf-8")


def write_toolchain_wrapper(destination: Path, toolchain: Path, target: str,
                            *, force: bool) -> None:
    """Create a Bazel repository that reads, but never edits, a vendor SDK."""
    if destination.exists():
        if not force:
            fail(f"toolchain repository exists: {destination} (use --force to replace it)")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    (destination / "sdk").symlink_to(toolchain, target_is_directory=True)
    build = (
        'package(default_visibility = ["//visibility:public"])\n\n'
        'filegroup(name = "gcc", srcs = ["sdk/bin/{target}-gcc"])\n'
        'filegroup(name = "ar", srcs = ["sdk/bin/{target}-ar"])\n'
        'filegroup(name = "ld", srcs = ["sdk/bin/{target}-ld"])\n'
        'filegroup(name = "nm", srcs = ["sdk/bin/{target}-nm"])\n'
        'filegroup(name = "objcopy", srcs = ["sdk/bin/{target}-objcopy"])\n'
        'filegroup(name = "objdump", srcs = ["sdk/bin/{target}-objdump"])\n'
        'filegroup(name = "strip", srcs = ["sdk/bin/{target}-strip"])\n'
        'filegroup(name = "as", srcs = ["sdk/bin/{target}-as"])\n\n'
        # Do not recursively glob a vendor SDK.  OpenWrt QSDK target trees
        # commonly contain compatibility symlink loops such as lib64 -> lib,
        # which causes Bazel analysis to fail before compilation.  The
        # compiler paths in cc_config.bzl are absolute QSDK paths and the
        # TensorFlow build uses standalone spawning, so declaring its tools is
        # sufficient and does not traverse the SDK filesystem.
        'filegroup(\n'
        '    name = "compiler_pieces",\n'
        '    srcs = [":ar", ":as", ":gcc", ":ld", ":nm", ":objcopy", ":objdump", ":strip"],\n'
        ')\n\n'
        'filegroup(\n'
        '    name = "compiler_components",\n'
        '    srcs = [":ar", ":as", ":gcc", ":ld", ":nm", ":objcopy", ":objdump", ":strip"],\n'
        ')\n'
    ).format(target=target)
    (destination / "BUILD.bazel").write_text(build, encoding="utf-8")
    (destination / "WORKSPACE").write_text(
        'workspace(name = "aarch64_linux_toolchain")\n', encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorflow-root", required=True,
                        help="TensorFlow v2.15 source checkout")
    parser.add_argument("--toolchain", required=True,
                        help="ARM64 cross-toolchain root")
    parser.add_argument("--compiler", default=None,
                        help="cross GCC inside --toolchain; defaults to "
                             "bin/aarch64-none-linux-gnu-gcc")
    parser.add_argument("--target-libc", choices=["glibc", "musl"], default="glibc",
                        help="target C library ABI; musl removes host glibc "
                             "headers from TensorFlow's embedded template")
    parser.add_argument("--sysroot", default=None,
                        help="target sysroot; required for musl targets")
    parser.add_argument("--staging-dir", default=None,
                        help="OpenWrt/QSDK staging_dir exported inside Bazel C/C++ actions; "
                             "defaults to the parent of --toolchain for musl")
    parser.add_argument("--output", required=True,
                        help="new local_config_embedded_arm repository directory")
    parser.add_argument("--toolchain-repository-output", default=None,
                        help="optional Bazel wrapper repository for an immutable "
                             "vendor toolchain")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output directory")
    args = parser.parse_args()

    tensorflow = Path(args.tensorflow_root).expanduser().resolve()
    toolchain = Path(args.toolchain).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    sysroot = Path(args.sysroot).expanduser().resolve() if args.sysroot else None
    staging_dir = (Path(args.staging_dir).expanduser().resolve()
                   if args.staging_dir else
                   (toolchain.parent if args.target_libc == "musl" else None))
    template = tensorflow / "tensorflow/tools/toolchains/embedded/arm-linux/cc_config.bzl.tpl"
    build_template = tensorflow / "tensorflow/tools/toolchains/embedded/arm-linux/aarch64-linux-toolchain.BUILD"
    compiler = (Path(args.compiler).expanduser().resolve()
                if args.compiler else toolchain / "bin/aarch64-none-linux-gnu-gcc")
    if not template.is_file() or not build_template.is_file():
        fail("TensorFlow embedded ARM toolchain templates were not found; use TensorFlow v2.15 source")
    if not compiler.is_file():
        fail(f"ARM64 compiler not found: {compiler}")
    if args.target_libc == "musl" and not sysroot:
        fail("--sysroot is required when --target-libc musl")
    if sysroot and not sysroot.is_dir():
        fail(f"sysroot directory not found: {sysroot}")
    if staging_dir and not staging_dir.is_dir():
        fail(f"staging directory not found: {staging_dir}")
    if output.exists():
        if not args.force:
            fail(f"output already exists: {output} (use --force to replace it)")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    version = compiler_version(compiler)
    target = compiler_target(compiler)
    config = template.read_text(encoding="utf-8")
    config = config.replace("%{AARCH64_COMPILER_PATH}%", str(toolchain))
    config = config.replace("%{PYTHON_INCLUDE_PATH}%", sysconfig_include())
    # ARMHF is deliberately absent from the generated BUILD.  Leave its
    # inactive branch syntactically valid without pretending ARMHF is tested.
    config = config.replace("%{ARMHF_COMPILER_PATH}%", "/opt/armhf-toolchain-not-configured")
    # TensorFlow v2.15 embeds both the GNU target name and GCC 11.3.1 in its
    # include and tool paths.  Replace them instead of asking a vendor SDK to
    # expose misleading aarch64-none-linux-gnu compatibility symlinks.
    config = config.replace("aarch64-none-linux-gnu", target)
    config = re.sub(rf"{re.escape(target)}/\d+\.\d+\.\d+",
                    f"{target}/{version}", config)
    config = re.sub(rf"{re.escape(target)}/include/c\+\+/\d+\.\d+\.\d+",
                    f"{target}/include/c++/{version}", config)
    if args.target_libc == "musl":
        # TensorFlow's stock embedded template unconditionally adds the host
        # /usr/include.  That mixes host glibc headers with OpenWrt musl
        # headers and breaks C++ compilation (for example __BEGIN_DECLS and
        # glibc-only strtoull_l).  The QSDK compiler already supplies its own
        # complete musl include search path.
        config = config.replace(
            '                                "-isystem",\n'
            '                                "/usr/include/",\n', '')
        config = config.replace(
            '                                "-isystem",\n'
            f'                                "{sysconfig_include()}",\n', '')
        # The same host path is also registered as a builtin include
        # directory later in the template, outside compile flags.
        config = config.replace('                "/usr/include",\n', '')
        config = config.replace(f'                "{sysconfig_include()}",\n', '')
        # OpenWrt compiler wrappers require STAGING_DIR. Bazel deliberately
        # sanitizes action environments, so a caller-side shell export (and,
        # with some exec transitions, even --action_env) is not sufficient.
        # Attach the variable to the C/C++ toolchain itself so target and exec
        # actions which select this toolchain receive it deterministically.
        environment_feature = (
            '    openwrt_staging_environment_feature = feature(\n'
            '        name = "openwrt_staging_environment",\n'
            '        enabled = True,\n'
            '        env_sets = [\n'
            '            env_set(\n'
            '                actions = all_compile_actions + all_link_actions,\n'
            '                env_entries = [\n'
            f'                    env_entry(key = "STAGING_DIR", value = {json.dumps(str(staging_dir))}),\n'
            '                ],\n'
            '            ),\n'
            '        ],\n'
            '    )\n')
        marker = '    dbg_feature = feature(name = "dbg")\n'
        if marker not in config:
            fail("cannot locate feature insertion point in TensorFlow toolchain template")
        config = config.replace(marker, marker + '\n' + environment_feature, 1)
        marker = '                dbg_feature,\n'
        if marker not in config:
            fail("cannot locate feature list in TensorFlow toolchain template")
        config = config.replace(
            marker,
            marker + '                openwrt_staging_environment_feature,\n',
            1)
    if sysroot:
        config = config.replace("builtin_sysroot = None",
                                f"builtin_sysroot = {json.dumps(str(sysroot))}")
        config = config.replace(
            f"{toolchain}/{target}/libc/usr/include/",
            f"{sysroot}/usr/include/")
    compiler_wrapper = None
    if args.target_libc == "musl":
        compiler_wrapper = output / "compiler_wrapper.sh"
        compiler_wrapper.write_text(
            "#!/bin/sh\n"
            f"export STAGING_DIR={shlex.quote(str(staging_dir))}\n"
            f"exec {shlex.quote(str(compiler))} \"$@\"\n",
            encoding="utf-8")
        compiler_wrapper.chmod(0o755)
        real_compiler_path = f'{toolchain}/bin/{target}-gcc'
        if real_compiler_path not in config:
            fail("cannot locate ARM64 GCC tool path in generated configuration")
        config = config.replace(real_compiler_path, str(compiler_wrapper))
    (output / "cc_config.bzl").write_text(config, encoding="utf-8")
    write_build(output / "BUILD.bazel",
                include_compiler_wrapper=compiler_wrapper is not None)
    (output / "WORKSPACE").write_text(
        'workspace(name = "local_config_embedded_arm")\n', encoding="utf-8")

    # A vendor SDK may be shared or read-only.  In that case create a separate
    # wrapper repository in the user's workspace instead of writing BUILD
    # metadata into the SDK itself.
    toolchain_repository = None
    if args.toolchain_repository_output:
        toolchain_repository = Path(args.toolchain_repository_output).expanduser().resolve()
        write_toolchain_wrapper(toolchain_repository, toolchain, target, force=args.force)
    else:
        # Backward-compatible mode for a developer-owned ARM GNU toolchain.
        # The separate wrapper mode above is preferred for vendor QSDKs.
        toolchain_build = toolchain / "BUILD.bazel"
        if not toolchain_build.exists() and not (toolchain / "BUILD").exists():
            build = build_template.read_text(encoding="utf-8")
            toolchain_build.write_text(
                build.replace("aarch64-none-linux-gnu", target), encoding="utf-8")
        workspace = toolchain / "WORKSPACE.bazel"
        if not workspace.exists() and not (toolchain / "WORKSPACE").exists():
            workspace.write_text('workspace(name = "aarch64_linux_toolchain")\n', encoding="utf-8")

    print("status=success")
    print(f"gcc_version={version}")
    print(f"target={target}")
    print(f"compiler={compiler}")
    print(f"config={output}")
    print(f"toolchain={toolchain}")
    if toolchain_repository:
        print(f"toolchain_repository={toolchain_repository}")
    if sysroot:
        print(f"sysroot={sysroot}")
    if staging_dir:
        print(f"staging_dir={staging_dir}")


def sysconfig_include() -> str:
    import sysconfig
    return sysconfig.get_paths()["include"]


if __name__ == "__main__":
    main()
