export type RespanParams = {
    custom_identifier?: string | number;
    customer_identifier?: string | number;
    customer_email?: string;
    customer_name?: string;
    evaluation_identifier?: string | number;
    thread_identifier?: string | number;
    trace_group_identifier?: string | number;
    group_identifier?: string | number;
    metadata?: Record<string, any>;
    properties?: Record<string, any>;
    prompt?: Record<string, any> | string;
    environment?: string;
}



