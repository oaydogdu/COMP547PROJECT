# Final Findings Template

## Research Question

When does randomized parallel decoding give meaningful speed gains without
unacceptable quality loss in small-scale autoregressive image generation?

## Setup

- Compute: Colab Pro (T4 default, stronger GPU only for critical runs)
- Datasets: Fashion-MNIST, CIFAR-10
- Conditions: sequential baseline + schedule/parallelism ablations

## Core Results

### Speed

- Latency trend:
- Throughput trend:

### Quality

- FID trend:
- Visual sample quality trend:

## Decision Boundaries

- Practical parallelism range:
- Where quality drop becomes unacceptable:
- Recommended schedule per dataset complexity:

## Final Recommendation (Cost/Performance)

- Default configuration:
- High-speed fallback:
- Conservative high-quality fallback:

## Risks and Limitations

- Small-scale limitation:
- FID sample-size limitation:
- Compute budget constraints:
