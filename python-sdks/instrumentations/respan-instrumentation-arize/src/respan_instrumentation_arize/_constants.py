"""Arize instrumentation constants."""

from __future__ import annotations

from dataclasses import dataclass

from respan_sdk.constants.span_attributes import RESPAN_METADATA

ARIZE_INSTRUMENTATION_NAME = "arize"

ARIZE_METADATA_INTEGRATION = f"{RESPAN_METADATA}.integration"
ARIZE_METADATA_RESOURCE = f"{RESPAN_METADATA}.arize_resource"
ARIZE_METADATA_OPERATION = f"{RESPAN_METADATA}.arize_operation"
ARIZE_METADATA_STATUS_CODE = f"{RESPAN_METADATA}.arize_status_code"


@dataclass(frozen=True)
class ArizeClientSpec:
    """One Arize SDK client class and the public methods to instrument."""

    module_name: str
    class_name: str
    resource: str
    methods: tuple[str, ...]


ARIZE_CLIENT_SPECS = (
    ArizeClientSpec(
        module_name="arize.spans.client",
        class_name="SpansClient",
        resource="spans",
        methods=(
            "delete",
            "list",
            "annotate",
            "log",
            "update_evaluations",
            "update_annotations",
            "update_metadata",
            "export_to_df",
            "export_to_parquet",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.ml.client",
        class_name="MLModelsClient",
        resource="ml",
        methods=("log_stream", "log", "export_to_df", "export_to_parquet"),
    ),
    ArizeClientSpec(
        module_name="arize.datasets.client",
        class_name="DatasetsClient",
        resource="datasets",
        methods=(
            "list",
            "create",
            "get",
            "delete",
            "update",
            "list_examples",
            "append_examples",
            "annotate_examples",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.experiments.client",
        class_name="ExperimentsClient",
        resource="experiments",
        methods=(
            "list",
            "create",
            "get",
            "delete",
            "list_runs",
            "append_runs",
            "annotate_runs",
            "run",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.prompts.client",
        class_name="PromptsClient",
        resource="prompts",
        methods=(
            "list",
            "create",
            "get",
            "get_version",
            "update",
            "delete",
            "list_versions",
            "create_version",
            "get_version_by_label",
            "set_labels",
            "delete_label",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.evaluators.client",
        class_name="EvaluatorsClient",
        resource="evaluators",
        methods=(
            "list",
            "get",
            "create_template_evaluator",
            "create_code_evaluator",
            "update",
            "delete",
            "list_versions",
            "get_version",
            "create_template_version",
            "create_code_version",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.projects.client",
        class_name="ProjectsClient",
        resource="projects",
        methods=("list", "create", "get", "delete", "update"),
    ),
    ArizeClientSpec(
        module_name="arize.spaces.client",
        class_name="SpacesClient",
        resource="spaces",
        methods=("list", "get", "create", "delete", "update", "add_user", "remove_user"),
    ),
    ArizeClientSpec(
        module_name="arize.annotation_configs.client",
        class_name="AnnotationConfigsClient",
        resource="annotation_configs",
        methods=("list", "create", "get", "delete"),
    ),
    ArizeClientSpec(
        module_name="arize.annotation_queues.client",
        class_name="AnnotationQueuesClient",
        resource="annotation_queues",
        methods=(
            "list",
            "get",
            "create",
            "update",
            "delete",
            "list_records",
            "add_records",
            "delete_records",
            "annotate_record",
            "assign_record",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.tasks.client",
        class_name="TasksClient",
        resource="tasks",
        methods=(
            "list",
            "get",
            "create_evaluation_task",
            "create_run_experiment_task",
            "update",
            "delete",
            "trigger_run",
            "list_runs",
            "get_run",
            "cancel_run",
            "wait_for_run",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.ai_integrations.client",
        class_name="AiIntegrationsClient",
        resource="ai_integrations",
        methods=("list", "get", "create", "update", "delete"),
    ),
    ArizeClientSpec(
        module_name="arize.api_keys.client",
        class_name="ApiKeysClient",
        resource="api_keys",
        methods=("list", "create", "create_service_key", "revoke", "refresh"),
    ),
    ArizeClientSpec(
        module_name="arize.organizations.client",
        class_name="OrganizationsClient",
        resource="organizations",
        methods=("list", "get", "create", "delete", "update", "add_user", "remove_user"),
    ),
    ArizeClientSpec(
        module_name="arize.users.client",
        class_name="UsersClient",
        resource="users",
        methods=(
            "list",
            "get",
            "create",
            "update",
            "delete",
            "resend_invitation",
            "bulk_delete",
            "reset_password",
        ),
    ),
    ArizeClientSpec(
        module_name="arize.roles.client",
        class_name="RolesClient",
        resource="roles",
        methods=("list", "get", "create", "update", "delete"),
    ),
    ArizeClientSpec(
        module_name="arize.role_bindings.client",
        class_name="RoleBindingsClient",
        resource="role_bindings",
        methods=("list", "create", "get", "update", "delete"),
    ),
    ArizeClientSpec(
        module_name="arize.resource_restrictions.client",
        class_name="ResourceRestrictionsClient",
        resource="resource_restrictions",
        methods=("restrict", "unrestrict"),
    ),
)
