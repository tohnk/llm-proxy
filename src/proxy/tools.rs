use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallAccumulator {
    pub index: usize,
    pub id: Option<String>,
    pub call_type: Option<String>,
    pub function_name: Option<String>,
    pub function_arguments: String,
}

#[derive(Debug, Default)]
pub struct ToolCallsBuffer {
    calls: BTreeMap<usize, ToolCallAccumulator>,
    /// Cap on the assembled arguments of any single call. A malfunctioning
    /// upstream can stream argument fragments indefinitely; nothing else bounds
    /// them, since they never reach `accumulated_text`.
    max_arguments_bytes: usize,
    truncated: bool,
}

impl ToolCallsBuffer {
    pub fn new(max_arguments_bytes: usize) -> Self {
        Self {
            calls: BTreeMap::new(),
            max_arguments_bytes,
            truncated: false,
        }
    }

    /// True once any call's arguments hit the cap; the assembled call is then
    /// certainly invalid JSON and must not be presented as a usable call.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    pub fn clear(&mut self) {
        self.calls.clear();
    }

    pub fn process_tool_call_deltas(&mut self, tool_calls: &[Value]) {
        for tc in tool_calls {
            let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let entry = self
                .calls
                .entry(index)
                .or_insert_with(|| ToolCallAccumulator {
                    index,
                    id: None,
                    call_type: None,
                    function_name: None,
                    function_arguments: String::new(),
                });

            if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                entry.id = Some(id.to_string());
            }
            if let Some(call_type) = tc.get("type").and_then(|t| t.as_str()) {
                entry.call_type = Some(call_type.to_string());
            }
            if let Some(func) = tc.get("function") {
                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                    let mut cur = entry.function_name.take().unwrap_or_default();
                    cur.push_str(name);
                    entry.function_name = Some(cur);
                }
                if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                    if entry.function_arguments.len().saturating_add(args.len())
                        > self.max_arguments_bytes
                    {
                        self.truncated = true;
                    } else {
                        entry.function_arguments.push_str(args);
                    }
                }
            }
        }
    }

    /// Build the downstream chunk representing all assembled tool calls.
    pub fn build_assembled_tool_calls_chunk(
        &self,
        downstream_id: &str,
        created: i64,
        model: &str,
        include_role: bool,
    ) -> Option<Value> {
        if self.calls.is_empty() {
            return None;
        }

        let mut tool_calls_arr = Vec::new();
        for entry in self.calls.values() {
            let tc_obj = serde_json::json!({
                "index": entry.index,
                "id": entry.id.as_deref().unwrap_or("call_0"),
                "type": entry.call_type.as_deref().unwrap_or("function"),
                "function": {
                    "name": entry.function_name.as_deref().unwrap_or(""),
                    "arguments": entry.function_arguments
                }
            });
            tool_calls_arr.push(tc_obj);
        }

        let mut delta = serde_json::Map::new();
        if include_role {
            delta.insert("role".to_string(), Value::String("assistant".to_string()));
        }
        delta.insert("tool_calls".to_string(), Value::Array(tool_calls_arr));

        Some(serde_json::json!({
            "id": downstream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": Value::Null
                }
            ]
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_buffering_and_assembly() {
        let mut buffer = ToolCallsBuffer::new(4 * 1024 * 1024);

        let delta1 = serde_json::json!([
            {
                "index": 0,
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup_stock",
                    "arguments": "{\"ticker\":"
                }
            }
        ]);
        buffer.process_tool_call_deltas(delta1.as_array().unwrap());

        let delta2 = serde_json::json!([
            {
                "index": 0,
                "function": {
                    "arguments": " \"AAPL\"}"
                }
            }
        ]);
        buffer.process_tool_call_deltas(delta2.as_array().unwrap());

        let assembled = buffer
            .build_assembled_tool_calls_chunk("id1", 100, "model1", true)
            .unwrap();
        assert_eq!(assembled["id"], "id1");
        assert_eq!(assembled["choices"][0]["delta"]["role"], "assistant");
        let tc = &assembled["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_123");
        assert_eq!(tc["function"]["name"], "lookup_stock");
        assert_eq!(tc["function"]["arguments"], "{\"ticker\": \"AAPL\"}");
    }
}
