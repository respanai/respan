import { PaginatedResponseType } from "./genericTypes.js";
import { MessageSchema } from "./logTypes.js";
import { 
    MessageRoleType, 
    ResponseFormatType, 
    ToolChoiceType, 
    ReasoningEffortType,
    DEFAULT_MODEL 
} from "../constants/promptConstants.js";
import { z } from "zod";

// Infer Message type from the schema in logTypes
export type Message = z.infer<typeof MessageSchema>;

export interface PromptVersion {
    id?: number | string;
    prompt_version_id?: string;
    description?: string;
    created_at?: string;
    updated_at?: string;
    version?: number;
    messages?: Message[];
    edited_by?: Record<string, any>;
    model?: string;
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    reasoning_effort?: ReasoningEffortType;
    variables?: Record<string, any>;
    readonly?: boolean;
    tools?: Record<string, any>[];
    tool_choice?: ToolChoiceType | Record<string, any>;
    response_format?: ResponseFormatType | Record<string, any>;
    json_schema?: Record<string, any>;
    is_enforcing_response_format?: boolean;
    prompt?: number | string; // Reference to parent prompt
    parent_prompt?: string; // Parent prompt ID
}

export interface Prompt {
    id?: number | string;
    name?: string;
    description?: string;
    prompt_id?: string;
    prompt_slug?: string;
    full_prompt_id?: string; // Same as prompt_id in responses
    current_version?: PromptVersion;
    live_version?: PromptVersion;
    prompt_versions?: PromptVersion[];
    prompt_activities?: Record<string, any>[];
    commit_count?: number;
    starred?: boolean;
    tags?: Record<string, any>[];
}

// Response types for different API endpoints
export interface PromptCreateResponse extends Prompt {}

export type PromptListResponse = PaginatedResponseType<Prompt>;

export interface PromptRetrieveResponse extends Prompt {}

export interface PromptVersionCreateResponse extends PromptVersion {}

export type PromptVersionListResponse = PaginatedResponseType<PromptVersion>;

export interface PromptVersionRetrieveResponse extends PromptVersion {}
