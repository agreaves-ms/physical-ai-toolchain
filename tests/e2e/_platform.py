"""Select the platform profile that the OSMO lifecycle e2e tests run against.

``PLATFORM_PROFILE`` selects ``azure`` (the default) or ``local``, as it does for the OSMO submit
scripts, and ``scripts/platform-profile.sh`` resolves it, so the tests and the scripts read one
source. The azure profile tracks runs and registers models in Azure ML. The local profile uses the
in-cluster MLflow server and model registry, trains on datasets in the data root, and builds policy
carriers into the local registry. ``AzurePlatform`` and ``LocalPlatform`` give the lifecycle tests one
interface to both.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.e2e._aml import AzureMLWorkspace, archive_all_model_versions, resolve_registered_model
from tests.e2e._common import RegisteredModel, format_command_failure, log_e2e, run_command
from tests.e2e._mlflow import (
    MLflowServer,
    delete_mlflow_registered_model,
    resolve_mlflow_registered_model,
)
from tests.e2e._osmo import OsmoLeRobotEvalPolicySource, osmo_lerobot_policy_source_from_model

PROFILE_VARIABLE = "PLATFORM_PROFILE"
AZURE_PROFILE = "azure"
LOCAL_PROFILE = "local"
PROFILE_NAMES = (AZURE_PROFILE, LOCAL_PROFILE)

_PROFILE_FIELDS = {
    "PROFILE_NAME": "name",
    "PROFILE_TRACKING_BACKEND": "tracking_backend",
    "PROFILE_OSMO_URL": "osmo_url",
    "PROFILE_OSMO_CONFIG_DIR": "osmo_config_dir",
    "PROFILE_MLFLOW_HOST_URI": "mlflow_host_uri",
    "PROFILE_DATASETS_HOST_DIR": "datasets_host_dir",
    "PROFILE_IMAGE_REGISTRY": "image_registry",
    "PROFILE_KUBECONFIG": "kubeconfig",
    "PROFILE_KUBE_CONTEXT": "kube_context",
}
_CARRIER_REFERENCE = re.compile(r"^Image \(digest\):\s+(\S+@sha256:[0-9a-f]{64})\s*$", re.MULTILINE)
_SHELL_VARIABLE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class PlatformProfileError(RuntimeError):
    """Raised when the platform profile cannot be selected or resolved."""


@dataclass(frozen=True)
class PlatformProfile:
    """The values of ``scripts/platform-profile.sh`` that the e2e fixtures read."""

    name: str
    tracking_backend: str
    osmo_url: str = ""
    osmo_config_dir: str = ""
    mlflow_host_uri: str = ""
    datasets_host_dir: str = ""
    image_registry: str = ""
    kubeconfig: str = ""
    kube_context: str = ""

    @property
    def is_local(self) -> bool:
        return self.name == LOCAL_PROFILE

    def profile_args(self) -> list[str]:
        """Return the option that selects this profile in a submit script."""
        return ["--profile", self.name]

    def client_environment(self) -> dict[str, str]:
        """Return the variables that point the OSMO CLI and kubectl at this profile's services.

        The azure profile keeps the caller's OSMO login and kubeconfig, so it sets none.
        """
        environment: dict[str, str] = {}
        if self.osmo_config_dir:
            environment["OSMO_CONFIG_FILE_DIR"] = self.osmo_config_dir
        if self.kubeconfig:
            environment["KUBECONFIG"] = self.kubeconfig
        return environment


def selected_profile_name(environ: Mapping[str, str] = os.environ) -> str:
    """Return the profile ``PLATFORM_PROFILE`` selects, which defaults to azure as in the submit scripts."""
    name = environ.get(PROFILE_VARIABLE, "").strip() or AZURE_PROFILE
    if name not in PROFILE_NAMES:
        raise PlatformProfileError(f"Unknown {PROFILE_VARIABLE} {name!r}: use {' or '.join(PROFILE_NAMES)}")
    return name


def profile_selected(name: str, environ: Mapping[str, str] = os.environ) -> bool:
    """Return whether ``PLATFORM_PROFILE`` selects ``name``, for collection-time skips; invalid values select none."""
    try:
        return selected_profile_name(environ) == name
    except PlatformProfileError:
        return False


def parse_profile(output: str) -> PlatformProfile:
    """Build a profile from the resolver's ``KEY=VALUE`` lines."""
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _PROFILE_FIELDS:
            values[_PROFILE_FIELDS[key]] = value
    if not values.get("name") or not values.get("tracking_backend"):
        raise PlatformProfileError(f"The platform profile resolver printed no profile:\n{output}")
    return PlatformProfile(**values)


def resolve_platform_profile(repo_root: Path, environ: Mapping[str, str] = os.environ) -> PlatformProfile:
    """Resolve the selected profile through ``scripts/platform-profile.sh``."""
    name = selected_profile_name(environ)
    result = run_command([str(repo_root / "scripts/platform-profile.sh"), "--profile", name], cwd=repo_root)
    if result.returncode != 0:
        raise PlatformProfileError(f"Cannot resolve the {name} platform profile\n\n{format_command_failure(result)}")
    return parse_profile(result.stdout)


def osmo_login_url(config_dir: str) -> str | None:
    """Return the service the OSMO CLI in ``config_dir`` is logged in to, or ``None`` before a login."""
    try:
        text = (Path(config_dir) / "login.yaml").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "url:":
            return fields[1]
    return None


def common_default(repo_root: Path, variable: str) -> str:
    """Return a pinned default, such as an image, from ``scripts/lib/common.sh``."""
    if not _SHELL_VARIABLE.fullmatch(variable):
        raise ValueError(f"Not a shell variable name: {variable!r}")
    result = run_command(
        ["bash", "-c", f'source scripts/lib/common.sh >/dev/null && printf "%s" "${{{variable}:-}}"'],
        cwd=repo_root,
    )
    if result.returncode != 0 or not result.stdout:
        raise AssertionError(f"scripts/lib/common.sh defines no {variable}\n\n{format_command_failure(result)}")
    return result.stdout


def parse_carrier_reference(output: str) -> str:
    """Return the digest-pinned image that ``build-model-image.sh`` reports in its summary."""
    match = _CARRIER_REFERENCE.search(output)
    if match is None:
        raise AssertionError(f"build-model-image.sh reported no digest-pinned image\n\n{output}")
    return match.group(1)


@dataclass(frozen=True)
class AzurePlatform:
    """Azure ML tracking and model registry, with datasets staged in Azure Blob Storage."""

    profile: PlatformProfile
    workspace: AzureMLWorkspace
    repo_root: Path

    @property
    def tracking(self) -> AzureMLWorkspace:
        return self.workspace

    def workspace_args(self) -> list[str]:
        """Return the Azure ML context that submit scripts pass to tracking in the job."""
        return [
            "--azure-subscription-id",
            self.workspace.subscription_id,
            "--azure-resource-group",
            self.workspace.resource_group,
            "--azure-workspace-name",
            self.workspace.workspace_name,
        ]

    def resolve_registered_model(self, model_name: str) -> RegisteredModel:
        return resolve_registered_model(self.repo_root, self.workspace, model_name=model_name)

    def remove_registered_model(self, model_name: str) -> None:
        archive_all_model_versions(self.repo_root, self.workspace, model_name)

    def lerobot_policy_source(self, model: RegisteredModel) -> OsmoLeRobotEvalPolicySource:
        return osmo_lerobot_policy_source_from_model(model)


@dataclass(frozen=True)
class LocalPlatform:
    """The in-cluster MLflow server and registry, data-root datasets, and the local image registry."""

    profile: PlatformProfile
    repo_root: Path

    @property
    def tracking(self) -> MLflowServer:
        return MLflowServer(self.profile.mlflow_host_uri)

    def workspace_args(self) -> list[str]:
        return []

    def resolve_registered_model(self, model_name: str) -> RegisteredModel:
        return resolve_mlflow_registered_model(self.tracking, model_name)

    def remove_registered_model(self, model_name: str) -> None:
        delete_mlflow_registered_model(self.tracking, model_name)

    def lerobot_policy_source(self, model: RegisteredModel) -> OsmoLeRobotEvalPolicySource:
        return OsmoLeRobotEvalPolicySource(
            args=("--policy-uri", model.uri),
            description=f"MLflow model {model.name}:{model.version}",
        )

    def build_policy_carrier(self, model: RegisteredModel, *, base_image: str) -> str:
        """Build a runnable carrier for ``model`` on ``base_image`` and return its digest-pinned reference."""
        log_e2e(f"Building a policy carrier for {model.name}:{model.version} on {base_image}")
        result = run_command(
            [
                str(self.repo_root / "fleet-deployment/setup/build-model-image.sh"),
                *self.profile.profile_args(),
                "--model-name",
                model.name,
                "--model-version",
                model.version,
                "--base-image",
                base_image,
            ],
            cwd=self.repo_root,
        )
        if result.returncode != 0:
            raise AssertionError(f"Policy carrier build failed\n\n{format_command_failure(result)}")
        reference = parse_carrier_reference(result.stdout)
        log_e2e(f"Built policy carrier {reference}")
        return reference

    def remove_policy_carrier(self, reference: str) -> None:
        """Delete a carrier's manifest from the local registry; its unshared layers go at the next registry GC."""
        repository, _, digest = reference.removeprefix(f"{self.profile.image_registry}/").partition("@")
        request = urllib.request.Request(
            f"http://{self.profile.image_registry}/v2/{repository}/manifests/{digest}", method="DELETE"
        )
        log_e2e(f"Deleting policy carrier {reference}")
        try:
            with urllib.request.urlopen(request, timeout=30):
                pass
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise AssertionError(f"Failed to delete policy carrier {reference}: HTTP {error.code}") from error


E2EPlatform = AzurePlatform | LocalPlatform


def mlflow_server_reachable(tracking_uri: str) -> bool:
    """Return whether an MLflow server answers its health check."""
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=10) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False
