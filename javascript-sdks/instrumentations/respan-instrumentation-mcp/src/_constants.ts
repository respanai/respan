import { RespanLogType } from "@respan/respan-sdk";

export const MCP_INSTRUMENTATION_NAME = "mcp";
export const MCP_INSTRUMENTATION_PACKAGE = "@respan/instrumentation-mcp";
export const MCP_LOG_METHOD_TS_TRACING = "ts_tracing";

export const MCP_METHOD_ATTRIBUTE = "mcp.method";
export const MCP_OPERATION_ATTRIBUTE = "mcp.operation";
export const MCP_TRANSPORT_ATTRIBUTE = "mcp.transport";

export type McpClientMethod =
  | "connect"
  | "listTools"
  | "callTool"
  | "listResources"
  | "readResource"
  | "listResourceTemplates"
  | "listPrompts"
  | "getPrompt";

export interface OperationConfig {
  entityName: string;
  logType: RespanLogType;
  operation: string;
  spanName: string;
}

export const CLIENT_OPERATION_CONFIG: Record<McpClientMethod, OperationConfig> =
  {
    connect: {
      entityName: "mcp.initialize",
      logType: RespanLogType.TASK,
      operation: "client.connect",
      spanName: "mcp.initialize",
    },
    listTools: {
      entityName: "mcp.list_tools",
      logType: RespanLogType.TASK,
      operation: "client.list_tools",
      spanName: "mcp.list_tools",
    },
    callTool: {
      entityName: "mcp.call_tool",
      logType: RespanLogType.TOOL,
      operation: "client.call_tool",
      spanName: "mcp.call_tool",
    },
    listResources: {
      entityName: "mcp.list_resources",
      logType: RespanLogType.TASK,
      operation: "client.list_resources",
      spanName: "mcp.list_resources",
    },
    readResource: {
      entityName: "mcp.read_resource",
      logType: RespanLogType.TASK,
      operation: "client.read_resource",
      spanName: "mcp.resource.read",
    },
    listResourceTemplates: {
      entityName: "mcp.list_resource_templates",
      logType: RespanLogType.TASK,
      operation: "client.list_resource_templates",
      spanName: "mcp.list_resource_templates",
    },
    listPrompts: {
      entityName: "mcp.list_prompts",
      logType: RespanLogType.TASK,
      operation: "client.list_prompts",
      spanName: "mcp.list_prompts",
    },
    getPrompt: {
      entityName: "mcp.get_prompt",
      logType: RespanLogType.TASK,
      operation: "client.get_prompt",
      spanName: "mcp.get_prompt",
    },
  };

export const SERVER_TOOL_OPERATION_CONFIG: OperationConfig = {
  entityName: "mcp.tool",
  logType: RespanLogType.TOOL,
  operation: "server.tool",
  spanName: "mcp.server.tool",
};
