"""Per-turn limits count every endpoint attempt, including cascade failures."""
import time


class AgentBudget:
    def __init__(self, seconds=180, output_tokens=8192, max_calls=14):
        if not 1 <= seconds <= 900 or not 256 <= output_tokens <= 65536 or not 1 <= max_calls <= 32:
            raise ValueError('智能体预算超出允许范围')
        self.deadline = time.monotonic() + seconds
        self.output_tokens = output_tokens
        self.max_calls = max_calls
        self.reserved = 0
        self.calls = []

    def reserve(self, maximum=2048):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or len(self.calls) >= self.max_calls or self.reserved >= self.output_tokens:
            raise RuntimeError('agent_turn_budget_exhausted')
        tokens = min(maximum, self.output_tokens - self.reserved)
        self.reserved += tokens
        record = {'max_output_tokens': tokens, 'status': 'started'}
        self.calls.append(record)
        return record, min(60, remaining), tokens

    def summary(self):
        usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        reported = 0
        for call in self.calls:
            if call.get('usage'):
                reported += 1
                for key in usage:
                    usage[key] += call['usage'].get(key, 0)
        return {'endpoint_calls': len(self.calls), 'reserved_max_output_tokens': self.reserved,
                'max_output_tokens': self.output_tokens, 'reported_usage': usage,
                'calls_with_reported_usage': reported,
                'remaining_seconds': max(0, round(self.deadline - time.monotonic(), 3)),
                'calls': self.calls}
