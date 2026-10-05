# Lab and verification standards

The labs are the spine of the book. They must teach causality rather than produce decorative output.

## Lab levels

- L0: deterministic CPU-only reasoning probe
- L1: local service or process experiment
- L2: containerized multi-process experiment
- L3: GPU experiment with explicit hardware requirements
- L4: cluster experiment, optional and cost-bounded
- L5: paper-scale reproduction, clearly labeled as partial when resources differ

Every lab declares level, hardware, software versions, expected duration, cost risk, cleanup, and fallback.

## Required evidence

Each lab records:

- command and environment
- input sizes and workload shape
- measurements with units
- baseline and intervention
- expected direction of change
- explanation of the mechanism
- failure interpretation
- reproducibility limits

A passing test proves only the asserted contract. It does not prove production readiness, scalability, or educational quality.
