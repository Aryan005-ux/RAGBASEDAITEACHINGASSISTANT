import time
import threading
from collections import deque
import numpy as np
from app.config import (
    SLA_LATENCY_MS,
    AWS_LAMBDA_COST_PER_MS,
    AWS_ECS_TASK_COST_PER_SEC
)

class CostTracker:
    def __init__(self, window_size: int = 10000):
        self.start_time = time.time()
        self.total_requests = 0
        self.sla_violations = 0
        self.total_lambda_cost = 0.0
        self.lock = threading.Lock()
        
        # Sliding window for latency calculation (to compute p99 and p95)
        self.latencies = deque(maxlen=window_size)
        
    def record_inference(self, latency_ms: float):
        with self.lock:
            self.total_requests += 1
            self.latencies.append(latency_ms)
            
            # Check SLA violation
            if latency_ms > SLA_LATENCY_MS:
                self.sla_violations += 1
                
            # Lambda cost tracking (based on duration of this request)
            # AWS Lambda rounds duration to nearest millisecond for billing
            billed_duration_ms = max(1.0, round(latency_ms))
            self.total_lambda_cost += billed_duration_ms * AWS_LAMBDA_COST_PER_MS

    def get_stats(self):
        with self.lock:
            uptime = time.time() - self.start_time
            reqs = self.total_requests
            violations = self.sla_violations
            lambda_cost = self.total_lambda_cost
            
            # Calculate latency percentiles
            lat_list = list(self.latencies)
            
        if not lat_list:
            avg_lat = 0.0
            p50 = 0.0
            p95 = 0.0
            p99 = 0.0
        else:
            avg_lat = float(np.mean(lat_list))
            p50 = float(np.percentile(lat_list, 50))
            p95 = float(np.percentile(lat_list, 95))
            p99 = float(np.percentile(lat_list, 99))
            
        sla_compliance = ((reqs - violations) / reqs * 100.0) if reqs > 0 else 100.0
        
        # Calculate ECS Fargate Cost
        # ECS runs continuously, so cost is uptime * rate
        ecs_cost = uptime * AWS_ECS_TASK_COST_PER_SEC
        
        # Costs per 1000 inferences
        lambda_cost_per_1000 = (lambda_cost / reqs * 1000.0) if reqs > 0 else 0.0
        ecs_cost_per_1000 = (ecs_cost / reqs * 1000.0) if reqs > 0 else 0.0
        
        # Current average throughput (RPS)
        current_rps = reqs / uptime if uptime > 0 else 0.0
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_inferences": reqs,
            "sla_limit_ms": SLA_LATENCY_MS,
            "sla_violations": violations,
            "sla_compliance_percentage": round(sla_compliance, 2),
            "latency_metrics_ms": {
                "avg": round(avg_lat, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2)
            },
            "throughput_rps": round(current_rps, 2),
            "costs_usd": {
                "lambda_total_cost": round(lambda_cost, 6),
                "lambda_cost_per_1000_inferences": round(lambda_cost_per_1000, 6),
                "ecs_total_cost": round(ecs_cost, 6),
                "ecs_cost_per_1000_inferences": round(ecs_cost_per_1000, 6)
            }
        }

# Global singleton instance
tracker = CostTracker()
