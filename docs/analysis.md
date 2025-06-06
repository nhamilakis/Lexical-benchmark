# Lexical Benchmark Metrics 


## Code Structure (`lexical_benchmark/metrics/`)

### Package Scripts

```
lexical_benchmark/metrics/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration classes and validation
├── loaders.py               # Data loading utilities
├── calculators.py           # Core metric calculation classes
├── processors.py            # Data processing and aggregation
└── state.py                 # CDI state management
```

### Executable Scripts

```
scripts/
├── compute_metrics.py       # Main metrics computation
└── test_cdi_thresholds.py   # CDI threshold testing
```





## File Structure (`lexical_benchmark/metrics/`)

```
generations/
├── checkpoints/
|       ├──dataset/lang/split/chunk/model/
            ├──generation_{hour_per_year}_{temperature}.obj  
            
└── months   # CDI threshold testing
        ├──dataset/lang/split/chunk/model/hour_per_year/{month}_{temperature}.txt  

```