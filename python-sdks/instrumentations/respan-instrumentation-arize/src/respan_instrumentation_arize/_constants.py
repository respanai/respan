"""Arize instrumentation constants."""

from __future__ import annotations

from dataclasses import dataclass

from respan_sdk.constants.span_attributes import RESPAN_METADATA

ARIZE_INSTRUMENTATION_NAME = "arize"

ARIZE_METADATA_INTEGRATION = f"{RESPAN_METADATA}.integration"
ARIZE_METADATA_RESOURCE = f"{RESPAN_METADATA}.arize_resource"
ARIZE_METADATA_OPERATION = f"{RESPAN_METADATA}.arize_operation"


@dataclass(frozen=True)
class ArizeClientSpec:
    """One Arize SDK client class and the public methods to instrument."""

    module_name: str
    class_name: str
    resource: str
    methods: tuple[str, ...]


ARIZE_CLIENT_SPECS = (
    ArizeClientSpec(
        "arize.ai_integrations.client",
        "AiIntegrationsClient",
        "ai_integrations",
        ("list", "get", "create", "update", "delete"),
    ),
    ArizeClientSpec(
        "arize.integrations.client",
        "IntegrationsClient",
        "integrations",
        (
            "create_agent",
            "create_llm",
            "delete",
            "get",
            "list",
            "update_agent",
            "update_llm",
        ),
    ),
    ArizeClientSpec(
        "arize.audit_logs.client", "AuditLogsClient", "audit_logs", ("list",)
    ),
    ArizeClientSpec(
        "arize.datasets.client",
        "DatasetsClient",
        "datasets",
        (
            "list",
            "create",
            "get",
            "delete",
            "update",
            "list_examples",
            "append_examples",
            "annotate_examples",
            "delete_examples",
            "update_examples",
        ),
    ),
    ArizeClientSpec(
        "arize.experiments.client",
        "ExperimentsClient",
        "experiments",
        (
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
        "arize.projects.client",
        "ProjectsClient",
        "projects",
        ("list", "create", "get", "delete", "update"),
    ),
    ArizeClientSpec(
        "arize.ml.client",
        "MLModelsClient",
        "ml",
        ("log_stream", "log", "export_to_df", "export_to_parquet"),
    ),
    ArizeClientSpec(
        "arize.organizations.client",
        "OrganizationsClient",
        "organizations",
        ("list", "get", "create", "delete", "update", "add_user", "remove_user"),
    ),
    ArizeClientSpec(
        "arize.spans.client",
        "SpansClient",
        "spans",
        (
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
    ArizeClientSpec("arize.traces.client", "TracesClient", "traces", ("list",)),
    ArizeClientSpec(
        "arize.annotation_configs.client",
        "AnnotationConfigsClient",
        "annotation_configs",
        (
            "list",
            "create",
            "get",
            "delete",
            "create_categorical",
            "create_continuous",
            "create_freeform",
            "update_categorical",
            "update_continuous",
            "update_freeform",
        ),
    ),
    ArizeClientSpec(
        "arize.annotation_queues.client",
        "AnnotationQueuesClient",
        "annotation_queues",
        (
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
        "arize.spaces.client",
        "SpacesClient",
        "spaces",
        ("list", "get", "create", "delete", "update", "add_user", "remove_user"),
    ),
    ArizeClientSpec(
        "arize.prompts.client",
        "PromptsClient",
        "prompts",
        (
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
        "arize.api_keys.client",
        "ApiKeysClient",
        "api_keys",
        ("list", "create", "create_service_key", "revoke", "refresh"),
    ),
    ArizeClientSpec(
        "arize.evaluators.client",
        "EvaluatorsClient",
        "evaluators",
        (
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
            "create_remote_evaluator",
            "create_remote_version",
            "delete_versions",
        ),
    ),
    ArizeClientSpec(
        "arize.resource_restrictions.client",
        "ResourceRestrictionsClient",
        "resource_restrictions",
        ("restrict", "unrestrict", "list"),
    ),
    ArizeClientSpec(
        "arize.role_bindings.client",
        "RoleBindingsClient",
        "role_bindings",
        ("list", "create", "get", "update", "delete"),
    ),
    ArizeClientSpec(
        "arize.roles.client",
        "RolesClient",
        "roles",
        ("list", "get", "create", "update", "delete"),
    ),
    ArizeClientSpec(
        "arize.tasks.client",
        "TasksClient",
        "tasks",
        (
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
        "arize.users.client",
        "UsersClient",
        "users",
        (
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
        "arize.webhooks.client",
        "WebhooksClient",
        "webhooks",
        (
            "create",
            "create_subscription",
            "delete",
            "delete_subscription",
            "get",
            "get_subscription",
            "list",
            "list_delivery_attempts",
            "list_subscriptions",
            "test",
            "update",
        ),
    ),
)
