/**
 * Frozen demo datasets for `respan demo`, hand-authored as typed TypeScript.
 *
 * Two datasets, numbered to match the demo prompt/evaluator they pair with:
 *   - "Demo dataset 1: support messages"  -> prompt 1 / evaluator 1
 *       Rows are customer messages that probe Lumen's refund/escalation policy
 *       edges. No `expected_output` — evaluator 1 scores compliance from the
 *       reply alone (it uses {{input}}/{{output}} only).
 *   - "Demo dataset 3: call transcripts"  -> prompt 3 / evaluator 3
 *       Rows are call transcripts, each with a gold-label classification in
 *       `expected_output` (evaluator 3 compares against {{expected_output}}).
 *
 * `input` is an object of the prompt's template variables (`customer_message` /
 * `call_transcript`); it is stored stringified and parsed into the row's mapped
 * fields. `expected_output` is authored here as an object for readability and
 * JSON-stringified at seed time — passing a raw object makes the platform coerce
 * it to single-quoted python-repr (invalid JSON the grader would then see).
 */

/** One dataset row: template variables plus an optional gold label. */
export interface DemoDatasetRow {
  /** The prompt's template variables for this row. */
  input: Record<string, unknown>;
  /** Gold label (object; stringified to JSON at seed time). */
  expected_output?: unknown;
}

/** One demo dataset: name/description plus its rows. */
export interface DemoDatasetFixture {
  name: string;
  description?: string;
  rows: DemoDatasetRow[];
}

export const DEMO_DATASETS: DemoDatasetFixture[] = [
  {
    name: 'Demo dataset 1: support messages',
    description:
      'Customer messages that probe Lumen Electronics refund/escalation policy edges, for evaluating the support agent (prompt 1).',
    rows: [
      {
        input: {
          customer_message:
            "Hi, my wireless earbuds (order A-50231) arrived two days ago but the left one won't charge. I'd like a refund, please.",
        },
      },
      {
        input: {
          customer_message:
            "I bought the clearance Lumen soundbar (order A-49120, it was marked final sale) last week and I've changed my mind. I'd like my money back.",
        },
      },
      {
        input: {
          customer_message:
            "Order A-51002 was supposed to arrive yesterday and it's still not here. I want a refund for the delay.",
        },
      },
      {
        input: {
          customer_message:
            'Tracking says my package (A-48817) was delivered three days ago, but I never received it. I want a full refund.',
        },
      },
      {
        input: {
          customer_message: 'Please process a refund on order A-52340 for $349.99.',
        },
      },
      {
        input: {
          customer_message:
            "My Lumen 4K monitor (order A-53001) stopped working after two days — it was $279. I'd like a refund because it's defective.",
        },
      },
      {
        input: {
          customer_message: "Where is my order A-50777? I just want to know when it'll arrive.",
        },
      },
    ],
  },
  {
    name: 'Demo dataset 3: call transcripts',
    description:
      'Support call transcripts with gold-label classifications, for evaluating the call-transcript classifier (prompt 3).',
    rows: [
      {
        input: {
          call_transcript:
            'Agent: Hi, this is Dana at Lumen Electronics support. How can I help you today?\n' +
            "Customer: My monitor, order A-51440, arrived with a cracked panel. I'm pretty annoyed.\n" +
            "Agent: I'm really sorry about that. I've created a replacement order — it ships today and you'll get tracking by email.\n" +
            'Customer: Oh, great. Thank you, that was easy.\n' +
            'Agent: My pleasure. Have a great day.',
        },
        expected_output: {
          primary_reason: 'product_defect',
          resolution_status: 'resolved',
          customer_sentiment_start: 'frustrated',
          customer_sentiment_end: 'positive',
          escalation_requested: false,
          followup_required: true,
          order_ids: ['A-51440'],
          products_mentioned: ['Lumen Monitor'],
          agent_name: 'Dana',
          summary:
            'Customer reported a cracked monitor panel (A-51440); agent created a same-day replacement shipment.',
          key_quote: 'My monitor, order A-51440, arrived with a cracked panel.',
          agent_followed_greeting: true,
          agent_empathy_shown: true,
          compliance_flags: [],
        },
      },
      {
        input: {
          call_transcript:
            'Agent: Thanks for calling Lumen Electronics, this is Marcus.\n' +
            "Customer: I was charged twice for order A-52210 and nobody's fixed it. I want a manager, now.\n" +
            "Agent: I can see two charges. I'm not able to reverse it from my side today.\n" +
            "Customer: Unacceptable. Escalate this or I'm calling my bank for a chargeback.\n" +
            "Agent: I'll escalate to a supervisor who will call you back within 24 hours.\n" +
            'Customer: Fine.',
        },
        expected_output: {
          primary_reason: 'billing_dispute',
          resolution_status: 'escalated',
          customer_sentiment_start: 'angry',
          customer_sentiment_end: 'frustrated',
          escalation_requested: true,
          followup_required: true,
          order_ids: ['A-52210'],
          products_mentioned: [],
          agent_name: 'Marcus',
          summary:
            'Customer disputed a double charge on A-52210 and demanded a manager; agent escalated to a supervisor with a 24-hour callback.',
          key_quote: "I was charged twice for order A-52210 and nobody's fixed it.",
          agent_followed_greeting: true,
          agent_empathy_shown: false,
          compliance_flags: ['no_identity_verification'],
        },
      },
      {
        input: {
          call_transcript:
            'Agent: Hi, my name is Max and this is the customer support line for Lumen Electronics. How can I help you today?\n' +
            'Customer: Finally. I ordered the Lumen Wireless Headphones almost two weeks ago, order A-48217, and they still have not shown up. This is ridiculous.\n' +
            "Agent: I'm really sorry about the wait, and I understand how frustrating that is. It looks like A-48217 was delayed at the carrier's sorting facility. I've flagged it and I'll email you an update within 24 hours.\n" +
            "Customer: Okay, I'd appreciate that.\n" +
            'Agent: Thank you for your patience.',
        },
        expected_output: {
          primary_reason: 'shipping_delay',
          resolution_status: 'partially_resolved',
          customer_sentiment_start: 'frustrated',
          customer_sentiment_end: 'neutral',
          escalation_requested: false,
          followup_required: true,
          order_ids: ['A-48217'],
          products_mentioned: ['Lumen Wireless Headphones'],
          agent_name: 'Max',
          summary:
            'Customer reported headphones order A-48217 was delayed; agent identified a carrier delay and promised an emailed update within 24 hours.',
          key_quote:
            'I ordered the Lumen Wireless Headphones almost two weeks ago and they still have not shown up.',
          agent_followed_greeting: true,
          agent_empathy_shown: true,
          compliance_flags: [],
        },
      },
      {
        input: {
          call_transcript:
            "Agent: Hello, you've reached Lumen Electronics, this is Priya.\n" +
            "Customer: Hi, I'd like to return my speaker, order A-50988. It works fine, I just don't need it.\n" +
            'Agent: No problem. It is within our 30-day window, so I have issued a full refund of $59.99 and emailed a prepaid return label.\n' +
            'Customer: Perfect, thanks so much.\n' +
            "Agent: You're welcome!",
        },
        expected_output: {
          primary_reason: 'return_or_refund',
          resolution_status: 'resolved',
          customer_sentiment_start: 'neutral',
          customer_sentiment_end: 'positive',
          escalation_requested: false,
          followup_required: true,
          order_ids: ['A-50988'],
          products_mentioned: ['Lumen Speaker'],
          agent_name: 'Priya',
          summary:
            'Customer requested a return for unwanted speaker A-50988; agent issued a full refund and sent a prepaid return label.',
          key_quote: "I'd like to return my speaker, order A-50988. It works fine, I just don't need it.",
          agent_followed_greeting: true,
          agent_empathy_shown: false,
          compliance_flags: [],
        },
      },
      {
        input: {
          call_transcript:
            "Customer: My Lumen smart hub won't connect to wifi no matter what I try.\n" +
            'Agent: Have you tried resetting it? Hold the button for ten seconds.\n' +
            'Customer: Yes, three times. Still nothing.\n' +
            "Agent: I'm not sure what else to suggest. I'll open a ticket and someone from tech will email you.\n" +
            'Customer: So you can’t actually help me right now?\n' +
            'Agent: Not today, sorry.',
        },
        expected_output: {
          primary_reason: 'technical_support',
          resolution_status: 'unresolved',
          customer_sentiment_start: 'frustrated',
          customer_sentiment_end: 'frustrated',
          escalation_requested: false,
          followup_required: true,
          order_ids: [],
          products_mentioned: ['Lumen Smart Hub'],
          agent_name: null,
          summary:
            'Customer’s smart hub would not connect to wifi; a basic reset failed and the agent opened a ticket for tech to follow up by email.',
          key_quote: "My Lumen smart hub won't connect to wifi no matter what I try.",
          agent_followed_greeting: false,
          agent_empathy_shown: false,
          compliance_flags: ['no_identity_verification', 'no_resolution_offered'],
        },
      },
    ],
  },
];
